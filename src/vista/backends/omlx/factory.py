"""Bind general tasks to Claude Code on this machine, served by oMLX."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vista.core.profiles import CompactionMode
from vista.core.runtime import TaskRuntime

from ..binding import (
    prepare_player,
    recovery_prompts,
    task_timeout as resolve_timeout,
    tool_interface,
)
from ..claude.preflight import run_runtime_preflight
from ..session import BackendSession, preflight, prepare_directory
from .runner import OmlxClaudeRunner


def create_backend(
    runtime: TaskRuntime,
    *,
    directory: Path,
    options: dict[str, Any],
    task_timeout: int | None = None,
) -> BackendSession:
    """``options`` come from ``runner.prepare``."""
    # The experiment's transport is Claude Code; oMLX is where it sends requests.
    preflight(runtime, "claude")
    visible, state, io = prepare_directory(directory)
    prepare_player(runtime, visible, io)
    runner = OmlxClaudeRunner(
        **options,
        visible_dir=visible,
        claude_config_dir=state,
        io_dir=io,
        controller=runtime.context,
        tool_interface=tool_interface(runtime, "claude"),
        recovery_prompts=recovery_prompts(runtime),
        model=runtime.experiment.model,
        effort=runtime.experiment.effort,
        native_compaction=runtime.experiment.profile.policy.compaction
        == CompactionMode.NATIVE,
        task_timeout=resolve_timeout(runtime, task_timeout),
    )

    def confirm_model() -> None:
        result, concrete = run_runtime_preflight(
            visible_dir=visible,
            config_dir=state / "preflight",
            io_dir=io / "preflight",
            claude_bin=runner.claude_bin,
            model=runner.model,
            effort=runner.effort,
            timeout=600,
            tool_interface=runner.tool_interface,
            native_compaction=runner.native_compaction,
            provider=runner.provider,
            provider_credential=runner.provider_credential,
            provider_base_url=runner.provider_base_url,
            max_output_tokens=runner.max_output_tokens,
            runner_class=OmlxClaudeRunner,
        )
        runner.expected_model_id = concrete
        runtime.artifacts.record(
            "model_preflight",
            {"resolved_model_id": concrete, "session_id": result.session_id},
        )

    return BackendSession(
        runtime, runner, kind="claude", before_first_run=confirm_model
    )
