"""Construct explicit task ports without replacing provider behavior."""

from __future__ import annotations

import json
from pathlib import Path

from vista.core.contracts import Effect
from vista.core.dispatcher import GameToolDispatcher
from vista.core.player import RecoveryPrompts, ToolInterface
from vista.core.profiles import CompactionMode
from vista.core.runtime import TaskRuntime
from vista.core.timeouts import bounded_task_timeout


def prepare_player(runtime: TaskRuntime, visible: Path, checkpoint_dir: Path) -> None:
    (visible / "screenshots").mkdir(mode=0o700)
    path = visible / "AGENTS.md"
    path.write_text(runtime.experiment.profile.prompts.system, encoding="utf-8")
    path.chmod(0o600)
    runtime.context.configure(visible, checkpoint_dir)


def tool_interface(runtime: TaskRuntime, kind: str) -> ToolInterface:
    dispatcher = GameToolDispatcher(
        controller=runtime.context,
        guide_path=runtime.context.guide_path,
        working_path=runtime.context.working_path,
        checkpoint_via_working=kind == "claude",
        delegate=runtime,
    )
    # A deciding session may gate tool calls (the decision cadence).
    gate = getattr(runtime.session, "gate", None)
    executor = gate(dispatcher) if callable(gate) else dispatcher

    def predict_images(name: str, arguments: object) -> int:
        spec = runtime.spec(name)
        if spec is None:
            return 0
        if (
            name == "inspect"
            and isinstance(arguments, dict)
            and isinstance(arguments.get("views"), list)
        ):
            return min(len(arguments["views"]), spec.max_images)
        return spec.max_images

    return ToolInterface(
        tuple(runtime.tools()),
        executor,
        frozenset(spec.name for spec in runtime.specs if spec.requires_ack),
        frozenset(
            spec.name
            for spec in runtime.specs
            if spec.effect == Effect.NONE and spec.max_images
        ),
        predict_images,
        runtime.acknowledge,
    )


def recovery_prompts(runtime: TaskRuntime) -> RecoveryPrompts:
    prompts = runtime.experiment.profile.prompts
    if runtime.experiment.profile.policy.compaction == CompactionMode.NATIVE:

        def native(_):
            return prompts.step or prompts.initial

        return RecoveryPrompts(native, native, native, native)

    def complete(recovery):
        working = recovery.working_memory
        provenance = getattr(recovery, "working_provenance", None)
        if working is not None and provenance is not None:
            working = {"saved_at": provenance, "content": working}
        return "\n".join(
            [
                "Environment record:",
                json.dumps(
                    {
                        "current_observation": recovery.current_observation,
                        "last_action_result": recovery.last_action_result,
                        "current_task_history": recovery.objective_history,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "",
                "Agent-authored notes:",
                json.dumps(
                    {"guide": recovery.guide, "working_memory": working},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "",
                prompts.recovery,
            ]
        )

    def resumed(recovery):
        return "\n".join(
            [
                "The player runtime restarted without changing the task state.",
                "Environment record:",
                json.dumps(
                    {
                        "current_observation": recovery.current_observation,
                        "last_action_result": recovery.last_action_result,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "",
                prompts.recovery,
            ]
        )

    return RecoveryPrompts(complete, complete, resumed, complete)


def task_timeout(runtime: TaskRuntime, override: int | None) -> int:
    return (
        override
        if override is not None
        else bounded_task_timeout(300, runtime.context.max_steps + 1)
    )
