"""Drive one AI GameStore run; shared backends own the context lifecycle."""

from __future__ import annotations

import importlib.metadata
import math
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vista.backends.preflight import source_fingerprint
from vista.backends.session import BackendSession
from vista.core.artifacts import ArtifactStore
from vista.core.contracts import ObservationBundle
from vista.core.json import canonical
from vista.core.profiles import Phase
from vista.core.runtime import BudgetExhausted, TaskRuntime
from vista.decide.integration import Decider, attach, enable

from .environment import (
    MAX_DECISIONS,
    PROTOCOL,
    GameStoreBrowser,
    GameStoreRuntime,
    PublicGame,
    load_human_median,
    normalized_score,
)
from .profiles import make_experiment, make_profile
from .session import SCREEN_PROMPT, GameStoreSession


def play_loop(
    session: GameStoreSession, runtime: TaskRuntime, backend: BackendSession
) -> None:
    started = False
    delivered: ObservationBundle | None = None
    redirect = ""
    try:
        while not session.finished and not runtime.terminal:
            phase = Phase.STEP if started else Phase.INITIAL
            blocks: tuple[str | ObservationBundle, ...] = ()
            public_text = redirect
            observation = (
                ObservationBundle()
                if phase == Phase.STEP and delivered == session.current
                else session.current
            )
            if redirect and observation.observations:
                blocks = (SCREEN_PROMPT, observation)
            if redirect:
                # The redirection precedes the screen so the label stays adjacent
                # to its image; the archived text keeps both.
                blocks = (redirect + "\n", *blocks)
            task_input = runtime.prepare_input(
                phase, observation=observation, public_text=public_text, blocks=blocks
            )
            before = session.decisions
            recorded = len(session.private)
            result = backend.run(task_input)
            started = True
            delivered = session.current
            failed = runtime.terminal or result.returncode != 0 or runtime.failure
            redirect = ""
            if not failed and session.decisions == before:
                reply = runtime.submit(result.final_message)
                if not runtime.terminal:
                    redirect = reply.text
            for record in session.private[recorded:]:
                runtime.artifacts.record("host_decision", dict(record))
            if failed:
                return
    except BudgetExhausted:
        return


def aggregate(
    results: list[dict[str, Any]], *, game_numbers: list[int], repeats: int
) -> dict[str, Any]:
    """Average raw scores over repeats per game, normalize, then geometric mean."""
    if (
        type(repeats) is not int
        or repeats < 1
        or not game_numbers
        or len(set(game_numbers)) != len(game_numbers)
    ):
        raise ValueError("Aggregation needs distinct games and a positive repeat count")
    expected = {
        (game, repeat) for game in game_numbers for repeat in range(1, repeats + 1)
    }
    actual = [(row["game_number"], row["repeat"]) for row in results]
    identities = {
        (
            row["protocol"],
            row["backend"],
            row["model"],
            row["effort"],
            row["frames_delivered"],
            row["win_policy"],
            row["max_decisions"],
        )
        for row in results
    }
    if len(identities) > 1:
        raise ValueError("Cannot aggregate unmatched experiments")
    complete = (
        len(actual) == len(set(actual))
        and set(actual) == expected
        and all(row["scored"] for row in results)
    )
    per_game: dict[str, float] = {}
    if complete:
        for game in game_numbers:
            rows = [row for row in results if row["game_number"] == game]
            medians = {row["human_median"] for row in rows}
            if len(medians) != 1:
                raise ValueError("Human references differ across repeats")
            per_game[str(game)] = normalized_score(
                sum(row["raw_score"] for row in rows) / repeats, medians.pop()
            )
    return {
        "planned": len(expected),
        "attempted": len(results),
        "scored": sum(bool(row["scored"]) for row in results),
        "complete": complete,
        "per_game_normalized_score": per_game,
        "geometric_mean_normalized": math.exp(
            sum(math.log(value) for value in per_game.values()) / len(per_game)
        )
        if per_game
        else None,
        "arithmetic_mean_normalized": sum(per_game.values()) / len(per_game)
        if per_game
        else None,
    }


def run_task(
    game: PublicGame,
    *,
    directory: Path,
    backend: str,
    model: str,
    effort: str,
    backend_factory: Callable[[TaskRuntime, Path], BackendSession],
    games_root: Path | None = None,
    repeat: int = 1,
    seed: int | None = None,
    headless: bool = True,
    max_decisions: int = MAX_DECISIONS,
    backend_segments: int | None = None,
    tool_calls: int | None = None,
    transport_provenance: dict[str, Any] | None = None,
    browser_factory: Callable[..., GameStoreRuntime] | None = None,
    human_median_loader: Callable[[str], float] = load_human_median,
    decider: Decider | None = None,
) -> dict[str, Any]:
    decider_summary = None
    if type(repeat) is not int or repeat < 1:
        raise ValueError("repeat must be a positive integer")
    source = "local" if games_root else "live"
    revision = game.revision
    root = directory.resolve()
    artifacts = ArtifactStore(root)
    artifacts.manifest(
        {
            "benchmark": "AI GameStore",
            "protocol": PROTOCOL,
            "benchmark_revision": revision,
            "game": game.manifest(),
            "repeat": repeat,
            "game_source": source,
            "games_root": str(games_root) if games_root is not None else None,
            "backend": backend,
            "model": model,
            "effort": effort,
        "frames_delivered": "five-segment-frames",
            "win_policy": "play-full-budget",
            "max_decisions": max_decisions,
            "seed": seed,
            "headless": headless,
            "transport": transport_provenance
            or {"validation": "caller-supplied-backend"},
            "source_files": source_fingerprint(),
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("playwright", "Pillow")
            },
        }
    )
    session = None
    runtime = None
    backend_session = None
    error = None
    result: dict[str, Any] = {}
    started = time.monotonic()
    try:
        browser = (browser_factory or GameStoreBrowser)(
            game, games_root=games_root, headless=headless, seed=seed
        )
        try:
            session = GameStoreSession(
                game,
                browser,
                max_decisions=max_decisions,
            )
        except BaseException:
            browser.close()
            raise
        profile = make_profile(
            game,
            backend=backend,
            max_decisions=max_decisions,
        )
        experiment = make_experiment(
            profile,
            game,
            backend=backend,
            model=model,
            effort=effort,
            max_decisions=max_decisions,
            seed=seed,
            source=source,
            revision=revision,
            backend_segments=backend_segments,
            tool_calls=tool_calls,
        )
        player, experiment = enable(session, experiment, decider)
        runtime = TaskRuntime(player, experiment, artifact_dir=root / "artifacts")
        decider_summary = attach(player, runtime)
        backend_session = backend_factory(runtime, root / "transport")
        play_loop(session, runtime, backend_session)
        result = session.result()
        error = runtime.failure
    except Exception:
        error = "run_exception"
        artifacts.record("private_failure", {"traceback": traceback.format_exc()})
        if session is not None:
            result = session.result()
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except Exception:
                error = "cleanup_failure"
                artifacts.record(
                    "private_failure", {"traceback": traceback.format_exc()}
                )
        elif session is not None:
            try:
                session.close()
            except Exception:
                error = "cleanup_failure"
    for key in (
        "decisions",
        "episodes",
        "finished",
        "won",
        "raw_score",
        "canvas",
        "browser_errors",
    ):
        result.setdefault(key, None)
    completed = bool(result.get("finished")) and error is None
    raw_score = result.get("raw_score")
    normalized = None
    human_median = None
    human_median_error = None
    if raw_score is not None:
        try:
            human_median = human_median_loader(game.game_id)
            normalized = normalized_score(raw_score, human_median)
        except Exception as exc:
            human_median_error = f"{type(exc).__name__}: {exc}"
    result.update(
        {
            "game_number": game.number,
            "game_id": game.game_id,
            "repeat": repeat,
            "benchmark_revision": revision,
            "game_source": source,
            "protocol": PROTOCOL,
            "backend": backend,
            "model": model,
            "effort": effort,
        "frames_delivered": "five-segment-frames",
            "win_policy": "play-full-budget",
            "max_decisions": max_decisions,
            "seed": seed,
            "completed": completed,
            "scored": completed and normalized is not None,
            "error": error,
            "human_median": human_median,
            "human_median_error": human_median_error,
            "normalized_score": normalized if completed else None,
            "usage": runtime.usage if runtime is not None else {},
            "checkpoints": runtime.context.checkpoints if runtime is not None else 0,
            "backend_usage": backend_session.usage_reports
            if backend_session is not None
            else [],
            "decider": decider_summary,
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    path = root / "result.json"
    path.write_text(canonical(result) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return result
