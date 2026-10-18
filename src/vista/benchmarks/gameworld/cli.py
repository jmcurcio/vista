"""Plan or execute GameWorld computer-use tasks with VISTA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vista.backends.launch import (
    add_backend_arguments,
    prepare_backend,
    transport_for,
    validate_backend_arguments,
)
from vista.core.json import thaw

from .preflight import check_transport
from .profiles import action_specs, make_experiment, make_profile
from .run import run_task
from .upstream import load_upstream


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("command", choices=("list", "plan", "run"))
    root.add_argument("--gameworld-root", type=Path, required=True)
    root.add_argument("--games-root", type=Path, required=True)
    root.add_argument("--game")
    root.add_argument("--task")
    add_backend_arguments(root, backends=("codex", "omlx"))
    root.add_argument("--model")
    root.add_argument("--effort")
    root.add_argument("--output", type=Path)
    root.add_argument("--seed", type=int)
    root.add_argument("--backend-segments", type=int)
    root.add_argument("--tool-calls", type=int)
    root.add_argument("--headed", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if any(value is not None and not value.strip() for value in (args.game, args.task)):
        cli.error("Game and task identifiers cannot be empty")
    if args.task and not args.game:
        cli.error("--task requires --game")
    if args.command != "list" and not all((args.backend, args.model, args.effort)):
        cli.error("plan/run require --backend, --model, and --effort")
    if any(
        value is not None and value < 1
        for value in (
            args.backend_segments,
            args.tool_calls,
            args.task_timeout,
        )
    ):
        cli.error("Runtime budgets and timeouts must be positive")
    validate_backend_arguments(args, cli.error)
    transport = transport_for(args.backend) if args.backend else None
    upstream = load_upstream(args.gameworld_root)
    games_root = args.games_root.expanduser().resolve(strict=True)
    games = [args.game] if args.game else upstream.catalog.list_games()
    tasks = []
    for game in games:
        for task in [args.task] if args.task else upstream.catalog.list_tasks(game):
            upstream.task(game, task)
            tasks.append((game, task))
    if args.command == "list":
        print(json.dumps({"games": len(games), "tasks": tasks}, indent=2))
        return 0
    plans: list[dict[str, Any]] = []
    for game_id, task_id in tasks:
        game, task = upstream.task(game_id, task_id)
        roles = []
        for index in range(game.role_count):
            profile = make_profile(
                upstream,
                game,
                task,
                index,
                backend=transport,
            )
            experiment = make_experiment(
                profile,
                backend=transport,
                model=args.model,
                effort=args.effort,
                width=game.width,
                height=game.height,
                seed=args.seed,
                backend_segments=args.backend_segments,
                tool_calls=args.tool_calls,
            )
            roles.append(
                {
                    **experiment.manifest(),
                    "action_tools": [
                        spec.public_definition()
                        for spec in action_specs(
                            upstream, game, index
                        )
                    ],
                }
            )
        plans.append(
            {
                "game_id": game_id,
                "task_id": task_id,
                "roles": roles,
            }
        )
    if args.command == "plan":
        print(
            json.dumps(
                {
                    "runs": plans,
                },
                indent=2,
            )
        )
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
    results = []
    for plan in plans:
        result = run_task(
            upstream,
            games_root=games_root,
            game_id=plan["game_id"],
            task_id=plan["task_id"],
            directory=args.output
            / f"{plan['game_id']}__{plan['task_id']}",
            backend=transport,
            model=args.model,
            effort=args.effort,
            backend_factory=backend_factory,
            seed=args.seed,
            headless=not args.headed,
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
                        "game_id",
                        "task_id",
                        "completed",
                        "scored",
                        "progress",
                        "success",
                        "error",
                    )
                }
            ),
            flush=True,
        )
        if not result["completed"]:
            break
    # Never silently drop incomplete planned runs.
    selected = results
    scored = [row for row in selected if row["scored"]]
    summary = [
            {
                "planned": len(tasks),
                "attempted": len(selected),
                "scored": len(scored),
                "complete": len(scored) == len(tasks),
                "mean_progress": sum(row["progress"] for row in scored) / len(scored)
                if len(scored) == len(tasks)
                else None,
                "success_rate": sum(row["success"] for row in scored) / len(scored)
                if len(scored) == len(tasks)
                else None,
            }
    ]
    summary_file = args.output / "summary.json"
    summary_file.write_text(
        json.dumps(thaw(summary), indent=2) + "\n", encoding="utf-8"
    )
    summary_file.chmod(0o600)
    return 0 if all(row["complete"] for row in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
