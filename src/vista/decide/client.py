"""A Jev SystemOne client: one wire protocol, interchangeable providers.

LocalJev (githubnext/localjev) implements the hosted Jev API's wire format, so
both are the same client pointed at different servers. Two providers answer
without a server and exist only as experimental controls: ``random`` (uniform
probabilities, a seeded uniform choice) and ``unavailable`` (always fails).

Every failure raises; nothing here substitutes a default decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import requests

RETRYABLE_STATUS = frozenset({500, 502, 503, 504, 529})
RETRY_DELAYS = (1.0, 4.0)


class DeciderError(RuntimeError):
    """The decision layer returned no usable decision."""


@dataclass(frozen=True)
class DeciderProvider:
    name: str
    base_url: str | None
    model: str
    credential_env: str | None
    max_options: int
    accepts_images: bool = False
    timeout_seconds: float = 180.0
    # What the server reported at readiness: recorded, never configured.
    server: dict[str, Any] | None = None

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


PROVIDERS = {
    # LocalJev's own limit is 128 choice options; its default port is 8080.
    "localjev": DeciderProvider(
        "localjev", "http://127.0.0.1:8080", "jev-latest", None, 128
    ),
    # The hosted API's address must be given explicitly (--decider-url).
    "jev": DeciderProvider("jev", None, "jev-latest", "JEV_API_KEY", 255),
    "random": DeciderProvider("random", None, "uniform", None, 255),
    "unavailable": DeciderProvider("unavailable", None, "none", None, 255),
}


def ready(provider: DeciderProvider) -> DeciderProvider:
    """Fail before a run starts if the provider cannot answer.

    Returns the provider with what the server reported, for provenance.
    """
    if provider.name in {"random", "unavailable"}:
        return provider
    if not provider.base_url:
        raise DeciderError(f"The {provider.name} provider needs --decider-url")
    if provider.name == "localjev":
        reply = requests.get(f"{provider.base_url}/ready", timeout=30)
        if reply.status_code != 200 or reply.json().get("status") != "ready":
            raise DeciderError(f"LocalJev is not ready: {reply.text[:300]}")
        return replace(provider, server=localjev_provenance(reply.json()))
    answer = decide(
        provider,
        state="Readiness check.",
        instructions="Is this a readiness check?",
        options={"yes": "Yes", "no": "No"},
    )
    return replace(provider, server={"model": answer["model"]})


def localjev_provenance(status: dict[str, Any]) -> dict[str, Any]:
    """The upstream model, its oMLX settings, and the default checkout's commit.

    The commit is read from local/localjev, where scripts/env_setup/localjev.sh
    installs it; a server started from elsewhere is recorded as unknown.
    """
    from vista.backends.omlx.runner import omlx_settings_snapshot

    checkout = Path(__file__).resolve().parents[3] / "local" / "localjev"
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    upstream = status.get("upstream_model")
    return {
        "upstream_model": upstream,
        "checkout_revision": revision or None,
        "omlx_settings": omlx_settings_snapshot(upstream) if upstream else None,
    }


def decide(
    provider: DeciderProvider,
    *,
    state: str,
    instructions: str,
    options: dict[str, str],
    sufficiency: str | None = None,
) -> dict[str, Any]:
    """One choice question, and optionally one noul question in the same call
    asking whether the evidence is sufficient; returns the decision and its
    accounting. `sufficient` is that probability, or None when not asked."""
    if not 2 <= len(options) <= provider.max_options:
        raise DeciderError(
            f"{provider.name} accepts 2 to {provider.max_options} options, "
            f"not {len(options)}"
        )
    started = time.monotonic()
    if provider.name == "unavailable":
        raise DeciderError("The decision layer is unavailable in this condition.")
    if provider.name == "random":
        request = json.dumps([state, instructions, options], sort_keys=True)
        seed = int(hashlib.sha256(request.encode()).hexdigest()[:16], 16)
        labels = list(options)
        return {
            "choice": random.Random(seed).choice(labels),
            "probabilities": {label: 1 / len(labels) for label in labels},
            "confidence": 0.0,
            # Uninformative, like the probabilities: never evidence of sufficiency.
            "sufficient": 0.5 if sufficiency else None,
            "model": "uniform",
            "seed": seed,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "attempts": 1,
            "latency_seconds": time.monotonic() - started,
        }
    if not provider.base_url:
        raise DeciderError(f"The {provider.name} provider needs --decider-url")
    headers = {"Content-Type": "application/json"}
    if provider.credential_env:
        credential = os.getenv(provider.credential_env)
        if not credential:
            raise DeciderError(f"{provider.credential_env} is not set")
        headers["Authorization"] = f"Bearer {credential}"
    body = {
        "model": provider.model,
        "state": state,
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": instructions,
                "criteria": options,
            }
        },
    }
    if sufficiency:
        body["questions"]["sufficient"] = {"type": "noul", "instructions": sufficiency}
    attempts = 0
    while True:
        attempts += 1
        try:
            reply = requests.post(
                f"{provider.base_url}/v1/systemone",
                json=body,
                headers=headers,
                timeout=provider.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise DeciderError(f"{provider.name} request failed: {exc}") from exc
        if reply.status_code in RETRYABLE_STATUS and attempts <= len(RETRY_DELAYS):
            time.sleep(RETRY_DELAYS[attempts - 1])
            continue
        break
    if reply.status_code != 200:
        raise DeciderError(
            f"{provider.name} returned HTTP {reply.status_code}: {reply.text[:300]}"
        )
    try:
        payload = reply.json()
        answer = payload["answers"]["decision"]
        probabilities = {
            label: float(answer["probabilities"][label]) for label in options
        }
        choice, certainty = answer["choice"], float(answer["confidence"])
        sufficient = (
            float(payload["answers"]["sufficient"]["noul"]) if sufficiency else None
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise DeciderError(f"{provider.name} returned a malformed decision") from exc
    if (
        answer.get("type") != "choice"
        or choice not in options
        or set(answer["probabilities"]) != set(options)
        or abs(sum(probabilities.values()) - 1) > 1e-3
        or not 0 <= certainty <= 1
        or (sufficient is not None and not 0 <= sufficient <= 1)
    ):
        raise DeciderError(f"{provider.name} returned an invalid decision")
    return {
        "choice": choice,
        "probabilities": probabilities,
        "confidence": certainty,
        "sufficient": sufficient,
        "model": payload.get("model"),
        "usage": payload.get("usage", {}),
        "attempts": attempts,
        "latency_seconds": time.monotonic() - started,
    }


def entropy_confidence(probabilities: list[float]) -> float:
    """1 minus normalized entropy, the confidence LocalJev reports."""
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return max(0.0, 1 - entropy / math.log(len(probabilities)))
