"""Run an ARC visual-game evaluation with Claude Code on this machine, served by oMLX.

The game, tools, prompt and lifecycle are the Claude harness's; only the
runtime differs (see vista.backends.omlx).
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from vista.backends.omlx.runner import OmlxClaudeRunner, add_arguments, prepare
from vista.benchmarks.arc3.claude.controller import write_json
from vista.benchmarks.arc3.claude.harness import build_parser as claude_parser, run_game
from vista.benchmarks.arc3.claude.runner import ClaudeCodeRunner as ArcClaudeRunner
from vista.benchmarks.arc3.run import load_run_environment, report_run_failure


class OmlxArcRunner(OmlxClaudeRunner, ArcClaudeRunner):
    """The host-process oMLX runtime with ARC's tools and recovery prompts."""


def build_parser() -> argparse.ArgumentParser:
    parser = claude_parser()
    parser.description = __doc__
    parser.set_defaults(model=None)
    parser.add_argument("--binary", type=Path, help="Claude Code (default: PATH)")
    add_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_run_environment()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.operation_mode != "offline" and not os.getenv("ARC_API_KEY"):
        parser.error("ARC_API_KEY is not set. Put it in .env or export it.")
    for name in ("max_steps", "timeout"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.max_invalid_retries < 0:
        parser.error("--max-invalid-retries must be >= 0")
    options, transport = prepare(args, parser.error)
    outcome = run_game(
        args, runner_factory=lambda **arc: OmlxArcRunner(**{**arc, **options})
    )
    manifest_path = outcome.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        runtime="omlx", transport=transport, claude_code_version=transport["version"]
    )
    write_json(manifest_path, manifest)
    print(f"Run directory: {outcome.run_dir}")
    print(f"Scorecard id: {outcome.scorecard_id}")
    print(f"Claude session id: {outcome.session_id}")
    if outcome.failure is not None:
        report_run_failure(outcome.run_dir, "claude")
        raise outcome.failure
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
