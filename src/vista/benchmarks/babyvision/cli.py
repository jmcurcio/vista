"""List, plan, run, judge or score BabyVision questions on one CLI backend."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from vista.backends.launch import (
    add_backend_arguments,
    prepare_backend,
    transport_for,
    validate_backend_arguments,
)
from vista.backends.preflight import check_transport

from .dataset import (
    BABYVISION_REVISION,
    DEFAULT_TASK_SET,
    TASK_SETS,
    Dataset,
    load_dataset,
)
from .profiles import make_experiment, make_profile
from .run import run_task
from .scoring import (
    aggregate,
    dump,
    exact_match,
    judge_answer,
    official_record,
    openai_chat,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument(
        "command", choices=("list", "plan", "run", "judge", "score", "rescore")
    )
    root.add_argument(
        "--babyvision-root",
        type=Path,
        default=os.environ.get("BABYVISION_ROOT"),
        help="Checkout of UniPat-AI/BabyVision at the pinned revision "
        "(default $BABYVISION_ROOT)",
    )
    root.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="Record the checkout's own revision instead of requiring the pin",
    )
    root.add_argument("--task-id", type=int, action="append", help="Repeatable")
    root.add_argument("--subtype", action="append", help="Repeatable official subtype")
    root.add_argument(
        "--task-set",
        choices=tuple(TASK_SETS),
        default=DEFAULT_TASK_SET,
        help="tracking48 (default): Maze, Connect the lines, Lines Observation; "
        "visual-tracking: the whole type (83); full: all 388",
    )
    add_backend_arguments(root, backends=("codex", "omlx"))
    root.add_argument("--model")
    root.add_argument("--effort")
    root.add_argument("--passes", type=int, default=1, help="Answers per question")
    root.add_argument("--output", type=Path)
    root.add_argument("--backend-segments", type=int)
    root.add_argument("--tool-calls", type=int)
    judge = root.add_argument_group("judge")
    judge.add_argument("--runs", type=Path, action="append", help="Run output dirs")
    judge.add_argument("--judge-base-url", help="OpenAI-compatible API root")
    judge.add_argument("--judge-model")
    judge.add_argument("--judge-key-env", default="JUDGE_API_KEY")
    return root


def _dataset(args: argparse.Namespace, error) -> Dataset:
    if args.babyvision_root is None:
        error("--babyvision-root (or BABYVISION_ROOT) is required")
    root = Path(args.babyvision_root)
    try:
        if not args.allow_unpinned:
            return load_dataset(root)
        from .dataset import checkout_revision

        revision = checkout_revision(root)
        if revision is None:
            # Not a checkout: the metadata hash is the only identity there is.
            unpinned = load_dataset(root, revision="unpinned")
            return load_dataset(root, revision=f"unpinned:{unpinned.fingerprint[:16]}")
        return load_dataset(root, revision=revision)
    except (OSError, RuntimeError, ValueError) as exc:
        error(str(exc))
        raise SystemExit(2) from exc


def _results(directories: list[Path]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for directory in directories:
        for path in sorted(directory.glob("**/result.json")):
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            row = json.loads(path.read_text(encoding="utf-8"))
            # The deterministic reading is cheap and may have improved since
            # the run; the recorded model output and key never change.
            if row.get("completed") and "ground_truth" in row and "answer_type" in row:
                row["exact_match"] = exact_match(
                    row.get("extracted_answer"),
                    row["ground_truth"],
                    answer_type=row["answer_type"],
                )
            row["_path"] = str(path)
            rows.append(row)
    return rows


def _planned_runs(directory: Path) -> set[tuple[int, int]]:
    """Recover the launch plan even when workers wrote no results."""
    if not directory.is_dir():
        raise ValueError(f"Run directory does not exist: {directory}")
    expected = set()
    for path in sorted(directory.rglob("batch.json")):
        batch = json.loads(path.read_text(encoding="utf-8"))
        if batch.get("benchmark") != "BabyVision":
            continue
        passes = batch["passes"]
        if not isinstance(passes, int) or passes < 1:
            raise ValueError(f"Invalid pass count in {path}")
        expected.update(
            (task, number)
            for task in batch["questions"]
            for number in range(1, passes + 1)
        )
    for path in sorted(directory.rglob("plan.json")):
        plan = json.loads(path.read_text(encoding="utf-8"))
        rows = plan.get("runs", []) if isinstance(plan, dict) else plan
        expected.update((row["task_id"], row["pass"]) for row in rows)
    return expected


def _score(directories: list[Path]) -> list:
    results = _results(directories)
    expected = set()
    for directory in directories:
        plan = _planned_runs(directory)
        actual = {
            (row["task_id"], row.get("pass", 1))
            for row in results
            if Path(row["_path"]).resolve().is_relative_to(directory.resolve())
        }
        if plan and not actual <= plan:
            raise ValueError(f"Results do not match the launch plan in {directory}")
        # Older outputs may predate saved plans; retain their observed cohort.
        expected.update(plan or actual)
    tasks = {task for task, _ in expected}
    passes = max((number for _, number in expected), default=1)
    if expected != {(task, p) for task in tasks for p in range(1, passes + 1)}:
        raise ValueError("Question sets differ across passes; score these runs separately")
    return _summary(results, expected=expected)


def _summary(
    results: list[dict[str, Any]], *, expected: set[tuple[int, int]]
) -> list:
    summary = aggregate(
        [{k: v for k, v in row.items() if k != "_path"} for row in results],
        planned=len({task for task, _ in expected}),
        passes=max((number for _, number in expected), default=1),
    )
    completed = {
        (row["task_id"], row.get("pass", 1))
        for row in results
        if row.get("completed")
    }
    summary["incomplete"] = sorted(
        f"{task}/p{number}" for task, number in expected - completed
    )
    return [summary]


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if args.command == "judge":
        return _judge(args, cli.error)
    if args.command == "rescore":
        # Persist recomputed exact-match scores for downstream readers.
        if not args.runs:
            cli.error("rescore requires --runs")
        changed = 0
        for row in _results(args.runs):
            path = Path(row.pop("_path"))
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored.get("exact_match") != row.get("exact_match"):
                stored["exact_match"] = row["exact_match"]
                path.write_text(dump(stored), encoding="utf-8")
                changed += 1
        print(json.dumps({"rescored": changed}))
        return 0
    if args.command == "score":
        if not args.runs:
            cli.error("score requires --runs")
        try:
            summary = _score(args.runs)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            cli.error(str(exc))
        print(dump(summary), end="")
        return 0
    dataset = _dataset(args, cli.error)
    try:
        questions = dataset.select(
            task_ids=args.task_id, subtypes=args.subtype, task_set=args.task_set
        )
    except ValueError as exc:
        cli.error(str(exc))
    if args.command == "list":
        print(
            json.dumps(
                {
                    "benchmark_revision": dataset.revision,
                    "metadata_sha256": dataset.fingerprint,
                    "pinned_revision": BABYVISION_REVISION,
                    "count": len(questions),
                    "questions": [question.manifest() for question in questions],
                },
                indent=2,
            )
        )
        return 0
    if not all((args.backend, args.model, args.effort)):
        cli.error("plan/run require --backend, --model, and --effort")
    if any(
        value is not None and value < 1
        for value in (
            args.passes,
            args.backend_segments,
            args.tool_calls,
            args.task_timeout,
        )
    ):
        cli.error("Passes, runtime budgets and timeouts must be positive")
    validate_backend_arguments(args, cli.error)
    transport = transport_for(args.backend)
    plans: list[dict[str, Any]] = []
    for question in questions:
        profile = make_profile(question, backend=transport)
        experiment = make_experiment(
            profile,
            question,
            backend=transport,
            model=args.model,
            effort=args.effort,
            revision=dataset.revision,
            image={"file": question.image.name},
            backend_segments=args.backend_segments,
            tool_calls=args.tool_calls,
        )
        for number in range(1, args.passes + 1):
            plans.append(
                {
                    "task_id": question.task_id,
                    "pass": number,
                    **experiment.manifest(),
                }
            )
    if args.command == "plan":
        print(json.dumps({"runs": plans}, indent=2))
        return 0
    if args.output is None:
        cli.error("run requires a new --output directory")
    backend_factory, provenance = prepare_backend(
        args, cli.error, check=check_transport
    )
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    plan_file = args.output / "plan.json"
    plan_file.write_text(json.dumps(plans, indent=2) + "\n", encoding="utf-8")
    plan_file.chmod(0o600)
    results: list[dict[str, Any]] = []
    failed = False
    expected = {(plan["task_id"], plan["pass"]) for plan in plans}
    for plan in plans:
        question = dataset.question(plan["task_id"])
        result = run_task(
            question,
            dataset.answers[question.task_id],
            directory=args.output
            / f"task{question.task_id}__p{plan['pass']}",
            backend=transport,
            model=args.model,
            effort=args.effort,
            backend_factory=backend_factory,
            revision=dataset.revision,
            fingerprint=dataset.fingerprint,
            pass_number=plan["pass"],
            backend_segments=args.backend_segments,
            tool_calls=args.tool_calls,
            transport_provenance=provenance,
        )
        results.append(result)
        print(
            json.dumps(
                {
                    key: result.get(key)
                    for key in (
                        "task_id",
                        "subtype",
                        "pass",
                        "completed",
                        "extracted_answer",
                        "exact_match",
                        "error",
                    )
                }
            ),
            flush=True,
        )
        summary_file = args.output / "summary.json"
        summary_file.write_text(
            dump(_summary(results, expected=expected)),
            encoding="utf-8",
        )
        summary_file.chmod(0o600)
        # A failed question is recorded and the suite goes on; the exit code
        # says so at the end. Incorrect answers are never rerun.
        failed = failed or not result["completed"]
    return 1 if failed else 0


def _judge(args: argparse.Namespace, error) -> int:
    """Apply the official judge to finished runs; the verdicts join their results."""
    if not args.runs:
        error("judge requires --runs")
    if not (args.judge_base_url and args.judge_model):
        error("judge requires --judge-base-url and --judge-model")
    key = os.environ.get(args.judge_key_env)
    if not key:
        error(f"judge requires the {args.judge_key_env} environment variable")
    complete = openai_chat(args.judge_base_url, key, args.judge_model)
    identity = {"base_url": args.judge_base_url, "model": args.judge_model}
    results = _results(args.runs)
    judged = 0
    for row in results:
        path = Path(row.pop("_path"))
        if row.get("judged") and row.get("judge") == identity:
            continue
        if not row.get("completed"):
            continue
        verdict = judge_answer(
            complete,
            question=row["question"],
            ground_truth=row["ground_truth"],
            extracted=row.get("extracted_answer"),
        )
        row.update(verdict, judged=True, judge=identity)
        path.write_text(dump(row), encoding="utf-8")
        judged += 1
    for directory in args.runs:
        rows = [
            {k: v for k, v in r.items() if k != "_path"} for r in _results([directory])
        ]
        for number in sorted({r.get("pass", 1) for r in rows}):
            official = [official_record(r) for r in rows if r.get("pass", 1) == number]
            out = directory / f"official_results_pass{number}.json"
            out.write_text(dump(official), encoding="utf-8")
    print(json.dumps({"judged": judged, "judge": identity}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
