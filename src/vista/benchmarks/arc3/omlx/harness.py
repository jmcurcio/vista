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
from typing import Any

from vista.backends.omlx.runner import OmlxClaudeRunner, add_arguments, prepare
from vista.benchmarks.arc3.claude.controller import write_json
from vista.benchmarks.arc3.claude.harness import build_parser as claude_parser, run_game
from vista.benchmarks.arc3.claude.recovery import TASK_OBJECTIVE
from vista.benchmarks.arc3.claude.runner import ClaudeCodeRunner as ArcClaudeRunner
from vista.benchmarks.arc3.run import load_run_environment, report_run_failure
from vista.core.session import append_jsonl
from vista.decide.integration import (
    Decider,
    DecisionGate,
    DecisionLayer,
    add_arguments as add_decider_arguments,
    decide_spec,
    resolve as resolve_decider,
)


class OmlxArcRunner(OmlxClaudeRunner, ArcClaudeRunner):
    """The host-process oMLX runtime with ARC's tools and recovery prompts,
    optionally with the `decide` tool (decisions logged to decisions.jsonl)."""

    def __init__(
        self,
        *,
        decider: Decider | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.decision_layer = None
        if decider is None:
            return
        log = self.io_dir / "decisions.jsonl"
        self.decision_layer = DecisionLayer(
            decider,
            TASK_OBJECTIVE,
            record=lambda event, value: append_jsonl(log, {"event": event, **value}),
        )
        # Inside the thermal gate, so decisions also wait out heat.
        gate = self.dispatcher
        gate._dispatcher = DecisionGate(
            gate._dispatcher, self.decision_layer, answers_decide=True
        )
        method = decider.method()
        if method not in self.agents_instructions:
            self.agents_instructions = f"{self.agents_instructions}\n\n{method}"
            # The file records what the player was given.
            self.agents_path.write_text(self.agents_instructions, encoding="utf-8")

    def _with_decide(self, definitions: Any) -> Any:
        if self.decision_layer is None:
            return definitions
        spec = decide_spec(self.decision_layer.decider.provider.max_options)
        return (*definitions, spec.public_definition())

    def _default_tool_definitions(self) -> Any:
        return self._with_decide(super()._default_tool_definitions())

    def _mcp_tool_definitions(self) -> Any:
        return self._with_decide(super()._mcp_tool_definitions())


def build_parser() -> argparse.ArgumentParser:
    parser = claude_parser()
    parser.description = __doc__
    parser.set_defaults(model=None)
    parser.add_argument("--binary", type=Path, help="Claude Code (default: PATH)")
    add_arguments(parser)
    add_decider_arguments(parser)
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
    decider = resolve_decider(args, parser.error)
    options, transport = prepare(args, parser.error)
    options.update(decider=decider)
    outcome = run_game(
        args, runner_factory=lambda **arc: OmlxArcRunner(**{**arc, **options})
    )
    manifest_path = outcome.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        runtime="omlx",
        transport=transport,
        claude_code_version=transport["version"],
        decider=None
        if decider is None
        else {**decider.manifest(), "log": "private/claude_io/decisions.jsonl"},
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
