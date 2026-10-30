"""Coordinate role inputs; shared backends own the context lifecycle."""

from __future__ import annotations

import importlib.metadata
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vista.backends.session import BackendSession
from vista.core.artifacts import ArtifactStore
from vista.core.contracts import ObservationBundle
from vista.core.json import canonical
from vista.core.profiles import Phase
from vista.core.runtime import BudgetExhausted, TaskRuntime
from vista.decide.integration import Decider, attach, enable

from .engine import GameWorldEngine, artifact_value
from .preflight import source_fingerprint
from .profiles import make_experiment, make_profile
from .session import RoleSession
from .upstream import (
    COMPUTER_USE_PROTOCOL,
    GAMES_REVISION,
    GAMEWORLD_REVISION,
    Upstream,
)


def play_loop(
    engine: GameWorldEngine,
    sessions: list[RoleSession],
    runtimes: list[TaskRuntime],
    backends: list[BackendSession],
) -> None:
    started: set[int] = set()
    try:
        while not engine.finished:
            index = engine.role
            session, runtime, backend = (
                sessions[index],
                runtimes[index],
                backends[index],
            )
            if runtime.terminal:
                return
            session.observe(engine.current)
            phase = Phase.INITIAL if index not in started else Phase.STEP
            observation = (
                ObservationBundle()
                if phase == Phase.STEP
                and session.delivered_current == engine.current
                else engine.current
            )
            task_input = runtime.prepare_input(
                phase,
                observation=observation,
                context_key=f"role-{index}",
            )
            before = len(engine.transitions)
            role_steps = [agent.step_index for agent in engine.agents]
            result = backend.run(task_input)
            started.add(index)
            if runtime.terminal and not engine.finished:
                return
            if result.returncode != 0 or runtime.failure:
                return
            current = [
                item
                for item in task_input.observation.observations
                if item.kind == "current"
            ]
            if current:
                session.delivered_current = ObservationBundle(tuple(current))
            if len(engine.transitions) != before:
                session.delivered_current = engine.current
            if len(engine.transitions) == before and not runtime.terminal:
                runtime.submit(result.final_message)
            for transition in engine.transitions[before:]:
                role_steps[transition.role] += 1
                runtime.artifacts.record(
                    "host_transition",
                    artifact_value(
                        {
                            "role": transition.role,
                            "action": transition.action,
                            "validity": transition.validity,
                            "evaluation": transition.evaluation,
                            "role_steps": list(role_steps),
                        }
                    ),
                )
    except BudgetExhausted:
        return


def run_task(
    upstream: Upstream,
    *,
    games_root: Path,
    game_id: str,
    task_id: str,
    directory: Path,
    backend: str,
    model: str,
    effort: str,
    backend_factory: Callable[[TaskRuntime, Path], BackendSession],
    seed: int | None = None,
    headless: bool = True,
    backend_segments: int | None = None,
    tool_calls: int | None = None,
    transport_provenance: dict[str, Any] | None = None,
    decider: Decider | None = None,
) -> dict[str, Any]:
    protocol = COMPUTER_USE_PROTOCOL
    decider_summaries: list[dict[str, Any] | None] = []
    root = directory.resolve()
    artifacts = ArtifactStore(root)
    artifacts.manifest(
        {
            "benchmark": "GameWorld",
            "protocol": protocol,
            "benchmark_revision": GAMEWORLD_REVISION,
            "games_revision": GAMES_REVISION,
            "game_id": game_id,
            "task_id": task_id,
            "backend": backend,
            "model": model,
            "effort": effort,
            "seed": seed,
            "headless": headless,
            "transport": transport_provenance
            or {"validation": "caller-supplied-backend"},
            "source_files": source_fingerprint(),
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in (
                    "playwright",
                    "Pillow",
                    "PyYAML",
                    "Jinja2",
                )
            },
        }
    )
    engine = None
    sessions: list[RoleSession] = []
    runtimes: list[TaskRuntime] = []
    backends = []
    error = None
    result: dict[str, Any] = {}
    started = time.monotonic()
    try:
        engine = GameWorldEngine(
            upstream,
            games_root=games_root,
            game_id=game_id,
            task_id=task_id,
            directory=root / "environment",
            seed=seed,
            headless=headless,
        )
        for index in range(engine.game.role_count):
            session = RoleSession(engine, index)
            sessions.append(session)
            profile = make_profile(
                upstream,
                engine.game,
                engine.task,
                index,
                backend=backend,
            )
            experiment = make_experiment(
                profile,
                backend=backend,
                model=model,
                effort=effort,
                width=engine.game.width,
                height=engine.game.height,
                seed=seed,
                backend_segments=backend_segments,
                tool_calls=tool_calls,
            )
            player, experiment = enable(session, experiment, decider)
            runtime = TaskRuntime(
                player, experiment, artifact_dir=root / f"role-{index}" / "artifacts"
            )
            decider_summaries.append(attach(player, runtime))
            runtimes.append(runtime)
            backends.append(
                backend_factory(runtime, root / f"role-{index}" / "transport")
            )
        play_loop(engine, sessions, runtimes, backends)
        result = engine.result()
        error = next((runtime.failure for runtime in runtimes if runtime.failure), None)
    except Exception:
        error = "run_exception"
        artifacts.record("private_failure", {"traceback": traceback.format_exc()})
        if engine is not None:
            result = engine.result()
    finally:
        for runtime in runtimes:
            try:
                runtime.close()
            except Exception:
                error = "role_cleanup_failure"
        if engine is not None:
            try:
                engine.close()
            except Exception:
                error = "environment_cleanup_failure"
                artifacts.record(
                    "private_failure", {"traceback": traceback.format_exc()}
                )
    completed = bool(result.get("finished")) and error is None
    evaluation = result.get("evaluation") or {}
    metrics = result.get("metrics") or {}
    scored = (
        completed
        and evaluation.get("status") in {"success", "fail"}
        and isinstance(metrics.get("progress"), (int, float))
        and not isinstance(metrics.get("progress"), bool)
        and 0 <= metrics["progress"] <= 1
        and not metrics.get("evaluation_config_errors")
        and not metrics.get("evaluation_runtime_issues")
    )
    result.update(
        {
            "game_id": game_id,
            "task_id": task_id,
            "protocol": protocol,
            "backend": backend,
            "model": model,
            "effort": effort,
            "seed": seed,
            "completed": completed,
            "scored": scored,
            "error": error,
            "progress": metrics.get("progress") if scored else None,
            "success": evaluation.get("status") == "success" if scored else None,
            "role_usage": [runtime.usage for runtime in runtimes],
            "role_checkpoints": [runtime.context.checkpoints for runtime in runtimes],
            "role_backend_usage": [backend.usage_reports for backend in backends],
            "role_deciders": decider_summaries,
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    path = root / "result.json"
    result = artifact_value(result)
    path.write_text(canonical(result) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return result
