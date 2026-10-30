"""Claude Code as a host process, with its model served by a local oMLX server.

The transport is the Claude backend's, unchanged: same CLI flags, MCP bridge,
compaction hooks and lifecycle. Only two things differ. Requests go to oMLX's
Anthropic-compatible ``/v1/messages``, and the CLI runs on this machine
instead of in the Linux player image, with the image's paths mapped to host
paths. The game tools are MCP-only either way (``--tools ""``); the container
was defense in depth, not part of the method.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import certifi
from dotenv import load_dotenv

from vista.backends.claude import runner as claude_runner
from vista.backends.claude.provider import ClaudeCodeProvider
from vista.backends.claude.runner import (
    CLAUDE_AUTO_COMPACT_PERCENT,
    CONTAINER_CLAUDE_BIN,
    CREDENTIAL_ENV_NAMES,
    MAX_MCP_OUTPUT_TOKENS,
    ClaudeCodeRunner,
    claude_version,
    provider_process_environment,
)

from .thermal import THERMAL_STATES, ThermalGate

OMLX_BASE_URL = "http://127.0.0.1:8000"
OMLX_KEY_ENV = "OMLX_API_KEY"
# Run from the installed package, as the image mounts them.
MCP_BRIDGE = Path(claude_runner.__file__).with_name("mcp_bridge.py").resolve()
COMPACT_HOOK = Path(claude_runner.__file__).with_name("compact_hook.py").resolve()
# The player image's fixed environment, plus what only a host process needs.
HOST_PLAYER_ENVIRONMENT = {
    "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB": "1",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
    "CLAUDE_CODE_DISABLE_CRON": "1",
    "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_AUTO_CONNECT_IDE": "false",
    "TERM": "dumb",
    "NO_COLOR": "1",
    "MAX_MCP_OUTPUT_TOKENS": str(MAX_MCP_OUTPUT_TOKENS),
    # The image's binary is read-only; a host install would update itself.
    "DISABLE_AUTOUPDATER": "1",
    # Local prefill and generation can outlast the default request timeout.
    "API_TIMEOUT_MS": "3000000",
}
# Inherited from the host, as a container would have them; nothing else is, so
# no CLAUDE_*/ANTHROPIC_* setting of the calling shell or IDE reaches the player.
HOST_ENVIRONMENT_ALLOWLIST = ("PATH", "USER", "LOGNAME", "SHELL", "LANG", "TMPDIR")
# Built into newer Claude Code than the release pins. The envelope check still
# rejects any plugin not disabled here.
# ponytail: per-version list; extend when an upgrade adds a builtin plugin.
BUILTIN_PLUGINS = ("cc-plugin-agents-md@builtin", "cc-plugin-plugin-authoring@builtin")


def omlx_provider(model: str, context_window: int) -> ClaudeCodeProvider:
    """``context_window`` must equal the model's oMLX ``max_context_window``."""
    return ClaudeCodeProvider(
        name="omlx",
        credential_env=OMLX_KEY_ENV,
        endpoint_protocol="anthropic-compatible",
        default_model=model,
        default_context_window=context_window,
        default_base_url=OMLX_BASE_URL,
        auto_compact_window=context_window,
        # Per turn, thinking included; oMLX's own thinking budget bounds that.
        max_output_tokens=16_384,
    )


def _without_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_secrets(item)
            for key, item in value.items()
            # Whole name parts, so max_tokens survives and api_key does not.
            if key != "auth"
            and not {"key", "token", "secret", "password"} & set(key.lower().split("_"))
        }
    return value


def omlx_settings_snapshot(model: str) -> dict[str, Any]:
    """The server and per-model settings in effect when the run starts.

    Thinking budget, sampling, MTP and memory guard live in oMLX, not in
    VISTA's experiment, yet they change results. Credentials are dropped.
    A change made mid-run is not captured.
    """
    base = Path(os.getenv("OMLX_BASE_PATH") or "~/.omlx").expanduser()
    snapshot: dict[str, Any] = {"base_path": str(base)}
    try:
        settings = json.loads((base / "settings.json").read_text(encoding="utf-8"))
        snapshot["server"] = _without_secrets(settings)
    except (OSError, ValueError) as exc:
        snapshot["server_error"] = f"{type(exc).__name__}: {exc}"
    try:
        models = json.loads((base / "model_settings.json").read_text(encoding="utf-8"))
        snapshot["model"] = _without_secrets(models.get("models", {}).get(model))
    except (OSError, ValueError) as exc:
        snapshot["model_error"] = f"{type(exc).__name__}: {exc}"
    if shutil.which("omlx"):
        version = subprocess.run(
            ["omlx", "--version"], capture_output=True, text=True, timeout=30
        )
        snapshot["version"] = (version.stdout or version.stderr).strip()
    return snapshot


def add_arguments(parser: Any) -> None:
    """The oMLX flags of the benchmark CLIs and the ARC harness."""
    parser.add_argument(
        "--context-window",
        type=int,
        help="omlx (required): the model's oMLX max_context_window, in tokens",
    )
    parser.add_argument("--omlx-base-url", default=OMLX_BASE_URL)
    parser.add_argument(
        "--thermal-pause-at",
        choices=(*THERMAL_STATES, "off"),
        default="serious",
        help="omlx: hold each tool call while macOS thermal state is at or "
        "above this (default: serious)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=0.0,
        help="omlx: seconds of idle before every tool call (default: 0)",
    )


def prepare(args: Any, error: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the oMLX flags; return runner options and run provenance.

    Provenance is taken here, before the run starts, and from here the Mac
    is kept from sleeping (lid-closed sleep on battery still happens): a
    sleeping Mac suspends Claude Code and oMLX, which ends the run.
    """
    load_dotenv(os.getenv("ARC3_ENV_FILE", ".env"))
    credential = os.getenv(OMLX_KEY_ENV)
    if not credential:
        error(f"omlx run requires {OMLX_KEY_ENV} (in .env or the environment)")
    if not args.model:
        error("omlx run requires --model, the oMLX model id")
    if not args.context_window or args.context_window < 1:
        error(
            "omlx run requires --context-window (the model's oMLX max_context_window)"
        )
    if args.cooldown < 0:
        error("--cooldown must be >= 0")
    # The versioned file, so a self-update of `claude` cannot change this run.
    binary = Path(args.binary or shutil.which("claude") or "").resolve()
    if not binary.is_file():
        error("Claude Code was not found; pass --binary")
    provider = omlx_provider(args.model, args.context_window)
    pause_at = THERMAL_STATES.get(args.thermal_pause_at)
    with binary.open("rb") as stream:
        binary_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    provenance = {
        "validation": "local-version-check",
        "version": claude_version(binary),
        "binary_sha256": binary_sha256,
        "provider": provider.name,
        "base_url": args.omlx_base_url,
        "context_window": args.context_window,
        "max_output_tokens": provider.max_output_tokens,
        "thermal_pause_at": pause_at,
        "cooldown_seconds": args.cooldown,
        "omlx_settings": omlx_settings_snapshot(args.model),
    }
    if shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(os.getpid())])
    options = dict(
        claude_bin=binary,
        provider=provider,
        provider_credential=credential,
        provider_base_url=args.omlx_base_url,
        auto_compact_window=args.context_window,
        auto_compact_percent=CLAUDE_AUTO_COMPACT_PERCENT,
        thermal_pause_at=pause_at,
        cooldown=args.cooldown,
    )
    return options, provenance


class OmlxClaudeRunner(ClaudeCodeRunner):
    def __init__(
        self,
        *,
        thermal_pause_at: int | None = THERMAL_STATES["serious"],
        cooldown: float = 0.0,
        **kwargs: Any,
    ) -> None:
        # The base checks a CA bundle exists; a host process uses the system's.
        kwargs["ca_bundle"] = kwargs.get("ca_bundle") or Path(certifi.where())
        super().__init__(**kwargs)
        self.dispatcher = ThermalGate(
            self.dispatcher,
            pause_at=thermal_pause_at,
            cooldown=cooldown,
            log=self.io_dir / "thermal.jsonl",
        )
        # Whatever is installed, pinned exactly for the life of this runner.
        self.claude_version_pin = claude_version(self.claude_bin)
        # An empty home and cwd outside any repository, like the image's
        # tmpfs. The cwd must stay fixed: Claude Code keys sessions by it.
        # ponytail: left in $TMPDIR (empty dirs); clean up if runs pile up.
        self.host_root = Path(tempfile.mkdtemp(prefix="vista-omlx-")).resolve()
        for name in ("home", "workspace"):
            (self.host_root / name).mkdir(mode=0o700)
        self._bridge_token: str | None = None

    def _process_cwd(self) -> Path:
        return self.host_root / "workspace"

    def _container_name(self, session_id: str) -> str | None:
        return None

    def _process_environment(self) -> dict[str, str]:
        environment = provider_process_environment(
            self.provider_environment,
            {
                name: value
                for name, value in os.environ.items()
                if name in HOST_ENVIRONMENT_ALLOWLIST or name.startswith("LC_")
            },
        )
        environment.update(HOST_PLAYER_ENVIRONMENT)
        environment.update(
            CLAUDE_CONFIG_DIR=str(self.claude_config_dir),
            HOME=str(self.host_root / "home"),
            # Claude Code cannot detect a local model's window.
            CLAUDE_CODE_MAX_CONTEXT_TOKENS=str(
                self.provider.context_window(self.model)
            ),
        )
        if self.auto_compact_window is not None:
            environment["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] = str(
                self.auto_compact_window
            )
            environment["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] = str(
                self.auto_compact_percent
            )
        if self.max_output_tokens is not None:
            environment["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(self.max_output_tokens)
        if self.mcp_tool_timeout_ms is not None:
            environment["MCP_TOOL_TIMEOUT"] = str(self.mcp_tool_timeout_ms)
        return environment

    def _write_mcp_config(self, path: Path, token: str) -> None:
        # The socket directory does not exist yet; the config is written with
        # the command, once it does.
        self._bridge_token = token

    def _docker_command(
        self,
        *,
        mcp_config_path: Path,
        settings_path: Path,
        socket_dir: Path,
        session_id: str,
        resume: bool,
        disable_slash_commands: bool = True,
    ) -> list[str]:
        """The image's Claude Code invocation, run directly on this machine."""
        assert self._bridge_token is not None
        self._write_host_mcp_config(
            mcp_config_path, self._bridge_token, socket_dir / "controller.sock"
        )
        container = super()._docker_command(
            mcp_config_path=mcp_config_path,
            settings_path=settings_path,
            socket_dir=socket_dir,
            session_id=session_id,
            resume=resume,
            disable_slash_commands=disable_slash_commands,
        )
        arguments = container[container.index(CONTAINER_CLAUDE_BIN) + 1 :]
        return [
            str(self.claude_bin),
            *(
                str(self.claude_config_dir / argument.removeprefix("/claude-config/"))
                if argument.startswith("/claude-config/")
                else argument
                for argument in arguments
            ),
        ]

    def _write_host_mcp_config(self, path: Path, token: str, socket: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "game": {
                            "type": "stdio",
                            "command": "/usr/bin/env",
                            "args": [
                                *(
                                    argument
                                    for name in CREDENTIAL_ENV_NAMES
                                    for argument in ("-u", name)
                                ),
                                sys.executable,
                                str(MCP_BRIDGE),
                            ],
                            "env": {
                                "ARC3_GAME_SOCKET": str(socket),
                                "ARC3_GAME_TOKEN": token,
                            },
                        }
                    }
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        path.chmod(0o600)

    def _container_io_path(self, path: Path) -> str:
        try:
            path.resolve().relative_to(self.io_dir)
        except ValueError as exc:
            raise ValueError(
                "Compact markers must live inside the Claude I/O dir."
            ) from exc
        return str(path.resolve())

    def _write_settings(self, path: Path, compact_event_log_name: str) -> None:
        super()._write_settings(path, compact_event_log_name)
        settings = json.loads(path.read_text(encoding="utf-8"))
        settings["enabledPlugins"] = dict.fromkeys(BUILTIN_PLUGINS, False)
        # The base hook names the image's paths; point it at this machine's.
        for entries in settings["hooks"].values():
            for entry in entries:
                for hook in entry["hooks"]:
                    words = shlex.split(hook["command"])
                    words = [
                        f"ARC3_COMPACT_EVENT_LOG={self.io_dir / compact_event_log_name}"
                        if word.startswith("ARC3_COMPACT_EVENT_LOG=")
                        else word
                        for word in words
                    ]
                    words[-2:] = [sys.executable, str(COMPACT_HOOK)]
                    hook["command"] = shlex.join(words)
        path.write_text(json.dumps(settings, separators=(",", ":")), encoding="utf-8")
        path.chmod(0o600)
