"""ARC3 defaults for the shared Claude transport."""

from pathlib import Path
from typing import Any

from vista.backends.claude import runner as _runner

from .recovery import (
    compact_recovery_prompt,
    fresh_runtime_recovery_prompt,
    retry_recovery_prompt,
)
from .tools import build_tools

DEFAULT_PINNED_CLAUDE_BIN = _runner.DEFAULT_PINNED_CLAUDE_BIN


class ClaudeCodeRunner(_runner.ClaudeCodeRunner):
    def _default_tool_definitions(self) -> tuple[dict[str, Any], ...]:
        return build_tools(self.display_size, include_compact_checkpoint=False)

    def _mcp_tool_definitions(
        self,
    ) -> tuple[dict[str, Any], ...] | list[dict[str, Any]]:
        if self.tool_interface is not None:
            return super()._mcp_tool_definitions()
        return build_tools(
            self.display_size,
            include_compact_checkpoint=False,
            reset_requires_retry_state=self.controller.reset_starts_fresh_session,
        )

    def _default_recovery_prompt(self, kind: str, recovery: Any) -> str:
        return {
            "compact": compact_recovery_prompt,
            "retry": retry_recovery_prompt,
            "fresh_runtime": fresh_runtime_recovery_prompt,
        }[kind](recovery)


def default_claude_bin() -> Path:
    return _runner.default_claude_bin(fallback=DEFAULT_PINNED_CLAUDE_BIN)


def validate_instruction_envelope(
    envelope: dict[str, Any] | None,
    *,
    display_size: int,
    allow_slash_commands: bool = False,
    tool_definitions: list[dict[str, Any]] | None = None,
    expected_version: str = _runner.PINNED_CLAUDE_VERSION,
) -> None:
    _runner.validate_instruction_envelope(
        envelope,
        display_size=display_size,
        allow_slash_commands=allow_slash_commands,
        expected_version=expected_version,
        tool_definitions=tool_definitions
        if tool_definitions is not None
        else build_tools(display_size, include_compact_checkpoint=False),
    )


def __getattr__(name: str) -> Any:
    return getattr(_runner, name)


__all__ = [name for name in dir(_runner) if not name.startswith("_")]
