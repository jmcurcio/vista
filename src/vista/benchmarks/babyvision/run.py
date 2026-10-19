"""Run one question end to end and score it on the host."""

from __future__ import annotations

import importlib.metadata
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from vista.backends.preflight import source_fingerprint
from vista.backends.session import BackendSession
from vista.core.artifacts import ArtifactStore
from vista.core.json import canonical
from vista.core.profiles import Phase
from vista.core.runtime import BudgetExhausted, TaskRuntime
from vista.decide.integration import Decider, attach, enable

from .dataset import Question, question_observation
from .official import extract_boxed_answer
from .profiles import PROTOCOL, make_experiment, make_profile
from .scoring import exact_match
from .session import BabyVisionSession


def answer_loop(
    session: BabyVisionSession, runtime: TaskRuntime, backend: BackendSession
) -> None:
    """One model run, then its final message is the answer.

    The official script sends one request and takes one reply. A CLI agent may
    take several model turns inside that one run, but it gets no second run:
    an empty final message is recorded as no answer, not re-prompted.
    """
    try:
        task_input = runtime.prepare_input(Phase.INITIAL)
        result = backend.run(task_input)
    except BudgetExhausted:
        return
    if runtime.terminal or result.returncode != 0 or runtime.failure:
        return
    runtime.submit(result.final_message)


def run_task(
    question: Question,
    ground_truth: str,
    *,
    directory: Path,
    backend: str,
    model: str,
    effort: str,
    backend_factory: Callable[[TaskRuntime, Path], BackendSession],
    revision: str,
    fingerprint: str | None = None,
    pass_number: int = 1,
    backend_segments: int | None = None,
    tool_calls: int | None = None,
    transport_provenance: dict[str, Any] | None = None,
    decider: Decider | None = None,
) -> dict[str, Any]:
    if type(pass_number) is not int or pass_number < 1:
        raise ValueError("pass_number must be a positive integer")
    root = directory.resolve()
    artifacts = ArtifactStore(root)
    observation = question_observation(question)
    visual = observation.observations[0].views[0]
    image = {
        "file": question.image.name,
        "width": visual.width,
        "height": visual.height,
        "mime_type": visual.mime_type,
        "sha256": visual.manifest()["sha256"],
    }
    artifacts.manifest(
        {
            "benchmark": "BabyVision",
            "track": "mllm",
            "protocol": PROTOCOL,
            "benchmark_revision": revision,
            "metadata_sha256": fingerprint,
            "question": question.manifest(),
            "image": image,
            "pass": pass_number,
            "backend": backend,
            "model": model,
            "effort": effort,
            "transport": transport_provenance
            or {"validation": "caller-supplied-backend"},
            "decider": decider.manifest() if decider is not None else None,
            "source_files": source_fingerprint(),
            "dependencies": {
                name: importlib.metadata.version(name) for name in ("Pillow", "regex")
            },
        }
    )
    session = None
    runtime = None
    backend_session = None
    error = None
    result: dict[str, Any] = {}
    decider_summary = None
    started = time.monotonic()
    try:
        session = BabyVisionSession(question, observation=observation)
        profile = make_profile(
            question, backend=backend
        )
        experiment = make_experiment(
            profile,
            question,
            backend=backend,
            model=model,
            effort=effort,
            revision=revision,
            image=image,
            backend_segments=backend_segments,
            tool_calls=tool_calls,
        )
        # The public choices, lettered as the question presents them.
        session, experiment = enable(
            session,
            experiment,
            decider,
            options={chr(65 + i): text for i, text in enumerate(question.options)}
            or None,
            # The bare question: the choices travel as options, and the answer
            # format instruction is for the agent only.
            objective=question.question,
        )
        runtime = TaskRuntime(session, experiment, artifact_dir=root / "artifacts")
        decider_summary = attach(session, runtime)
        backend_session = backend_factory(runtime, root / "transport")
        answer_loop(session, runtime, backend_session)
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
            session.close()
    for key in ("question", "answered", "raw_output"):
        result.setdefault(key, None)
    completed = bool(result.get("answered")) and error is None
    extracted = extract_boxed_answer(result.get("raw_output")) if completed else None
    result.update(
        {
            "task_id": question.task_id,
            "type": question.category,
            "subtype": question.subtype,
            "answer_type": question.answer_type,
            "pass": pass_number,
            "benchmark_revision": revision,
            "protocol": PROTOCOL,
            "backend": backend,
            "model": model,
            "effort": effort,
            "completed": completed,
            "error": error,
            "extracted_answer": extracted,
            # Host-side only: the run directory is never mounted for the player.
            "ground_truth": ground_truth,
            "exact_match": exact_match(
                extracted, ground_truth, answer_type=question.answer_type
            ),
            "judged": False,
            "llm_judge_result": None,
            "usage": runtime.usage if runtime is not None else {},
            "backend_usage": backend_session.usage_reports
            if backend_session is not None
            else [],
            "elapsed_seconds": time.monotonic() - started,
            "decider": decider_summary,
        }
    )
    path = root / "result.json"
    path.write_text(canonical(result) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return result
