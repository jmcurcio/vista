"""Shared command-line selection of a transport.

`--backend` names what the benchmark CLIs launch: `codex`, `claude`, or
`omlx` (the claude transport as a host process, served by a local oMLX).
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vista.backends.preflight import check_transport

BACKENDS = ("codex", "claude", "omlx")
TRANSPORTS = {"codex": "codex", "claude": "claude", "omlx": "claude"}


def transport_for(backend: str) -> str:
    try:
        return TRANSPORTS[backend]
    except KeyError:
        raise ValueError(f"Unknown backend {backend!r}") from None


def add_backend_arguments(
    parser: argparse.ArgumentParser, *, backends: tuple[str, ...] = BACKENDS
) -> None:
    # Imported here so naming an argument does not pull in a transport.
    from vista.backends.codex.runner import default_codex_auth_file

    parser.add_argument("--backend", choices=backends)
    parser.add_argument(
        "--auth-file",
        type=Path,
        # All benchmarks default to the file written by `codex login`.
        default=default_codex_auth_file(),
        help="Codex auth.json (default: ~/.codex/auth.json)",
    )
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--ca-bundle", type=Path)
    parser.add_argument("--task-timeout", type=int)
    if "omlx" in backends:
        from vista.backends.omlx.runner import add_arguments

        add_arguments(parser)
    if "claude" not in backends:
        return
    parser.add_argument("--claude-token-env", default="CLAUDE_CODE_OAUTH_TOKEN")
    parser.add_argument("--expected-model-id")
    parser.add_argument(
        "--compact-window",
        type=int,
        help="claude: auto-compaction window in tokens (default: provider or "
        "transport setting)",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        help="claude: CLAUDE_CODE_MAX_OUTPUT_TOKENS",
    )


def validate_backend_arguments(
    args: argparse.Namespace, error: Callable[[str], Any]
) -> None:
    for name in ("compact_window", "max_output_tokens"):
        value = getattr(args, name, None)
        if value is not None and value < 1:
            error(f"--{name.replace('_', '-')} must be positive")


def prepare_backend(
    args: argparse.Namespace,
    error: Callable[[str], Any],
    *,
    check: Callable[[str, Path | None], tuple[Path, dict[str, Any]]] = check_transport,
) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Validate credentials and the transport, then return (factory, provenance).

    Credential lookups happen before the transport check so a missing token is
    reported without touching Docker. Nothing here runs a model.
    """
    backend = args.backend
    transport = transport_for(backend)
    if backend == "omlx":
        return _prepare_omlx(args, error)
    if backend == "codex":
        if args.auth_file is None or not args.auth_file.is_file():
            error("Codex run requires an existing --auth-file")
        from vista.backends.codex import create_backend

        binary, provenance = check(transport, args.binary)

        def factory(runtime, directory):
            return create_backend(
                runtime,
                directory=directory,
                auth_file=args.auth_file,
                binary=binary,
                ca_bundle=args.ca_bundle,
                task_timeout=args.task_timeout,
                image=provenance["container_image_id"],
            )

        return factory, provenance
    token = os.environ.get(args.claude_token_env)
    if not token:
        error(
            f"Claude run requires the {args.claude_token_env} environment variable"
        )
    from vista.backends.claude import create_backend

    binary, provenance = check(transport, args.binary)

    def factory(runtime, directory):
        return create_backend(
            runtime,
            directory=directory,
            oauth_token=token,
            binary=binary,
            ca_bundle=args.ca_bundle,
            expected_model_id=args.expected_model_id,
            task_timeout=args.task_timeout,
            image=provenance["container_image_id"],
            auto_compact_window=args.compact_window,
            max_output_tokens=args.max_output_tokens,
        )

    return factory, provenance


def _prepare_omlx(
    args: argparse.Namespace, error: Callable[[str], Any]
) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Claude Code on this machine with oMLX as its model; no Docker, no OAuth."""
    from vista.backends.omlx import create_backend
    from vista.backends.omlx.runner import prepare

    options, provenance = prepare(args, error)

    def factory(runtime, directory):
        return create_backend(
            runtime,
            directory=directory,
            options=options,
            task_timeout=args.task_timeout,
        )

    return factory, provenance
