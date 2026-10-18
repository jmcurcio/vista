"""Confirm a native Claude runtime with the selected task tool surface disabled."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from vista.core.dispatcher import ToolExecution
from vista.core.player import ToolInterface

from .provider import ANTHROPIC_PROVIDER, ClaudeCodeProvider
from .quota import ClaudeRateLimitEvent
from .runner import (
    PLAYER_IMAGE,
    RUNTIME_PREFLIGHT_PROMPT,
    ClaudeCodeRunner,
    ClaudePreflightController,
    ClaudeResult,
    resolved_model_id,
    validate_instruction_envelope,
)


class _PreflightExecutor:
    def execute(self, tool: str, arguments: Any, call_id: str) -> ToolExecution:
        return ToolExecution(
            "Task tools are unavailable during runtime preflight.", False
        )


def run_runtime_preflight(
    *,
    visible_dir: Path,
    config_dir: Path,
    io_dir: Path,
    claude_bin: Path,
    model: str,
    effort: str,
    timeout: int,
    tool_interface: ToolInterface,
    oauth_token: str | None = None,
    rate_limit_observer: Callable[[ClaudeRateLimitEvent], None] | None = None,
    ca_bundle: Path | None = None,
    image: str = PLAYER_IMAGE,
    native_compaction: bool = False,
    provider: ClaudeCodeProvider = ANTHROPIC_PROVIDER,
    provider_credential: str | None = None,
    provider_base_url: str | None = None,
    max_output_tokens: int | None = None,
    runner_class: type[ClaudeCodeRunner] = ClaudeCodeRunner,
) -> tuple[ClaudeResult, str]:
    tools = replace(
        tool_interface,
        executor=_PreflightExecutor(),
        acknowledge=lambda call_id: None,
        predict_images=lambda name, arguments: 0,
    )
    runner = runner_class(
        visible_dir=visible_dir,
        claude_config_dir=config_dir,
        io_dir=io_dir,
        controller=ClaudePreflightController(),
        tool_interface=tools,
        claude_bin=claude_bin,
        ca_bundle=ca_bundle,
        model=model,
        effort=effort,
        oauth_token=oauth_token,
        task_timeout=max(60, timeout),
        rate_limit_observer=rate_limit_observer,
        image=image,
        native_compaction=native_compaction,
        provider=provider,
        provider_credential=provider_credential,
        provider_base_url=provider_base_url,
        max_output_tokens=max_output_tokens,
    )
    result = runner.run_task(RUNTIME_PREFLIGHT_PROMPT)
    if result.returncode != 0:
        raise RuntimeError(
            f"Claude runtime preflight failed; stderr: {result.stderr_path}"
        )
    validate_instruction_envelope(
        result.init_envelope,
        display_size=runner.display_size,
        tool_definitions=list(tools.definitions),
        expected_version=runner.claude_version_pin,
    )
    concrete_model_id = resolved_model_id(result)
    if result.final_message.strip() != "READY":
        raise RuntimeError("Claude runtime preflight returned an unexpected result.")
    return result, concrete_model_id
