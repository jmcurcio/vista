"""Decision-layer metrics from raw BabyVision run directories.

    uv run python -m vista.decide.analyze runs/NAME [runs/OTHER ...]

Reads only what runs recorded: each task's `result.json` (outcome, host-side
ground truth) and the `decision` events in `artifacts/events.jsonl`. Prints one
JSON object with three parts:

- tasks: outcome, time, tool calls and decider accounting per task;
- decisions: one row per returned decision, scored against the ground truth;
- summary: decider accuracy, calibration (ECE, Brier) of its top choice, and
  echo, i.e. agreement with the agent's withheld `leaning`, overall and when
  that leaning was wrong.

Nothing is sampled or re-run, so the numbers regenerate exactly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ECE_BINS = 15


def _events(task: Path) -> list[dict[str, Any]]:
    path = task / "artifacts" / "events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def expected_calibration_error(rows: list[tuple[float, bool]]) -> float | None:
    """Equal-mass bins over (top probability, correct)."""
    if not rows:
        return None
    rows = sorted(rows)
    size = max(1, len(rows) // ECE_BINS)
    bins = [rows[i : i + size] for i in range(0, len(rows), size)]
    return sum(
        len(b)
        / len(rows)
        * abs(sum(p for p, _ in b) / len(b) - sum(c for _, c in b) / len(b))
        for b in bins
    )


def analyze(directories: list[Path]) -> dict[str, Any]:
    tasks, decisions = [], []
    for directory in directories:
        for result_path in sorted(directory.glob("**/result.json")):
            task = result_path.parent
            result = json.loads(result_path.read_text(encoding="utf-8"))
            truth = result.get("ground_truth")
            events = _events(task)
            manifest_path = task / "artifacts" / "manifest.json"
            manifest = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.is_file()
                else {}
            )
            decider = manifest.get("presentation", {}).get("decider") or {}
            tasks.append(
                {
                    "task": str(task),
                    "profile": manifest.get("profile", {}).get("id"),
                    "prompt_version": decider.get("prompt_version"),
                    "completed": result.get("completed"),
                    "answer": result.get("extracted_answer"),
                    "truth": truth,
                    "correct": result.get("exact_match"),
                    "elapsed_seconds": result.get("elapsed_seconds"),
                    "tool_calls": result.get("usage", {}).get("tool_calls"),
                    "decider": result.get("decider"),
                    "gate_refusals": sum(e["event"] == "decision_gate" for e in events),
                }
            )
            for event in events:
                if event["event"] != "decision" or event.get("outcome") != "returned":
                    continue
                probabilities = event["probabilities"]
                leaning = event.get("leaning")
                decisions.append(
                    {
                        "task": str(task),
                        "choice": event["choice"],
                        "top_probability": max(probabilities.values()),
                        "confidence": event["confidence"],
                        "sufficient": event.get("sufficient"),
                        "verdict": (event.get("policy") or {}).get("verdict"),
                        "correct": event["choice"] == truth,
                        "leaning": leaning,
                        "agrees_with_leaning": None
                        if leaning is None
                        else event["choice"] == leaning,
                        "leaning_correct": None
                        if leaning is None
                        else leaning == truth,
                        "flagged_findings": len(event.get("flags") or []),
                        "latency_seconds": event.get("latency_seconds"),
                    }
                )
    with_leaning = [d for d in decisions if d["leaning"] is not None]
    wrong_leaning = [d for d in with_leaning if not d["leaning_correct"]]
    calibration = [(d["top_probability"], d["correct"]) for d in decisions]

    def rate(rows: list[dict[str, Any]], key: str) -> float | None:
        return sum(bool(r[key]) for r in rows) / len(rows) if rows else None

    return {
        "tasks": tasks,
        "decisions": decisions,
        "summary": {
            "tasks": len(tasks),
            "task_accuracy": rate(tasks, "correct"),
            "decisions": len(decisions),
            "decider_accuracy": rate(decisions, "correct"),
            "ece_top_choice": expected_calibration_error(calibration),
            "brier_top_choice": sum((p - c) ** 2 for p, c in calibration)
            / len(calibration)
            if calibration
            else None,
            "echo_rate": rate(with_leaning, "agrees_with_leaning"),
            "echo_rate_when_leaning_wrong": rate(wrong_leaning, "agrees_with_leaning"),
            "decisions_with_leaning": len(with_leaning),
            "commit_verdicts": sum(
                d["verdict"] in {"commit", "final"} for d in decisions
            ),
            "commit_accuracy": rate(
                [d for d in decisions if d["verdict"] in {"commit", "final"}], "correct"
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    paths = [Path(a) for a in (argv if argv is not None else sys.argv[1:])]
    if not paths:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    print(json.dumps(analyze(paths), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
