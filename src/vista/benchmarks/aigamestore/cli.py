"""List, plan or run AI GameStore games on one CLI backend."""

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
from vista.backends.preflight import check_transport
from vista.core.json import thaw

from .environment import MAX_DECISIONS, PUBLIC_GAMES, PublicGame, load_public_game
from .profiles import action_specs, make_experiment, make_profile
from .run import aggregate, run_task


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("command", choices=("list", "plan", "run"))
    root.add_argument(
        "--game",
        type=int,
        action="append",
        help="Public game number 1..10; repeatable. Omit to select all ten.",
    )
    root.add_argument(
        "--games-root",
        type=Path,
        help="Serve local game copies (gameN/index.html) instead of aigamestore.org",
    )
    add_backend_arguments(root, backends=("codex", "omlx"))
    root.add_argument("--model")
    root.add_argument("--effort")
    root.add_argument("--max-decisions", type=int, default=MAX_DECISIONS)
    root.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Independent runs per game",
    )
    root.add_argument("--output", type=Path)
    root.add_argument("--seed", type=int)
    root.add_argument("--backend-segments", type=int)
    root.add_argument("--tool-calls", type=int)
    root.add_argument("--headed", action="store_true")
    return root


def _select_games(games: list[int]) -> list[PublicGame]:
    return [load_public_game(number) for number in games]


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    games = args.game or list(PUBLIC_GAMES)
    if any(number not in PUBLIC_GAMES for number in games) or len(set(games)) != len(
        games
    ):
        cli.error("--game must name distinct public games in 1..10")
    if not 1 <= args.max_decisions <= MAX_DECISIONS:
        cli.error(f"--max-decisions must be in 1..{MAX_DECISIONS}")
    if args.command != "list" and not all((args.backend, args.model, args.effort)):
        cli.error("plan/run require --backend, --model, and --effort")
    if any(
        value is not None and value < 1
        for value in (
            args.repeats,
            args.backend_segments,
            args.tool_calls,
            args.task_timeout,
        )
    ):
        cli.error("Repeats, runtime budgets and timeouts must be positive")
    validate_backend_arguments(args, cli.error)
    transport = transport_for(args.backend) if args.backend else None
    if args.games_root is not None and not args.games_root.is_dir():
        cli.error("--games-root must be an existing directory")
    public = _select_games(games)
    if args.command == "list":
        print(json.dumps({"games": [game.manifest() for game in public]}, indent=2))
        return 0
    source = "local" if args.games_root is not None else "live"
    plans: list[dict[str, Any]] = []
    for game in public:
        profile = make_profile(
            game,
            backend=transport,
            max_decisions=args.max_decisions,
        )
        experiment = make_experiment(
            profile,
            game,
            backend=transport,
            model=args.model,
            effort=args.effort,
            max_decisions=args.max_decisions,
            seed=args.seed,
            source=source,
            backend_segments=args.backend_segments,
            tool_calls=args.tool_calls,
        )
        for repeat in range(1, args.repeats + 1):
            plans.append(
                {
                    "game_number": game.number,
                    "game_id": game.game_id,
                    "repeat": repeat,
                    **experiment.manifest(),
                    "action_tools": [
                        spec.public_definition() for spec in action_specs()
                    ],
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
    by_number = {game.number: game for game in public}
    results = []
    for plan in plans:
        number = plan["game_number"]
        result = run_task(
            by_number[number],
            directory=args.output
            / f"game{number}__r{plan['repeat']}",
            backend=transport,
            model=args.model,
            effort=args.effort,
            backend_factory=backend_factory,
            games_root=args.games_root,
            repeat=plan["repeat"],
            seed=args.seed,
            headless=not args.headed,
            max_decisions=args.max_decisions,
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
                        "game_number",
                        "game_id",
                        "repeat",
                        "completed",
                        "scored",
                        "decisions",
                        "episodes",
                        "raw_score",
                        "normalized_score",
                        "error",
                    )
                }
            ),
            flush=True,
        )
        # Never silently drop incomplete runs.
        summary = [
            aggregate(
                results,
                game_numbers=[game.number for game in public],
                repeats=args.repeats,
            )
        ]
        summary_file = args.output / "summary.json"
        summary_file.write_text(
            json.dumps(thaw(summary), indent=2) + "\n", encoding="utf-8"
        )
        summary_file.chmod(0o600)
        if not result["completed"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
