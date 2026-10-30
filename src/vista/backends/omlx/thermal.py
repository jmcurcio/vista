"""Pause the agent between model turns while the Mac runs hot.

Every model turn ends in a tool call, and Claude Code waits for its result,
so holding a tool call idles the model and lets the machine cool. The games
are paused or turn-based between actions, so the wait changes no task state;
it only adds wall-clock time, logged so timing analysis can subtract it.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import sys
import time
from functools import cache
from pathlib import Path
from typing import Any

# NSProcessInfo.thermalState, the level macOS itself throttles by.
THERMAL_STATES = {"nominal": 0, "fair": 1, "serious": 2, "critical": 3}
POLL_SECONDS = 30


@cache
def _process_info() -> tuple[Any, Any, Any] | None:
    if sys.platform != "darwin":
        return None
    ctypes.cdll.LoadLibrary(
        "/System/Library/Frameworks/Foundation.framework/Foundation"
    )
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.sel_registerName.restype = ctypes.c_void_p
    send = objc.objc_msgSend
    send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    send.restype = ctypes.c_void_p
    info = send(
        objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo")
    )
    return send, info, objc.sel_registerName(b"thermalState")


def thermal_state() -> int:
    """0 nominal, 1 fair, 2 serious, 3 critical; 0 where macOS cannot say."""
    found = _process_info()
    if found is None:
        return 0
    send, info, selector = found
    send.restype = ctypes.c_long
    return int(send(info, selector))


class ThermalGate:
    """A tool dispatcher that first waits out heat, then an optional cooldown.

    ``pause_at`` is a thermal state; a call waits until the state drops below
    it. ``cooldown`` seconds are added before every call regardless, lowering
    the model's duty cycle.
    """

    def __init__(
        self, dispatcher: Any, *, pause_at: int | None, cooldown: float, log: Path
    ) -> None:
        self._dispatcher = dispatcher
        self.pause_at = pause_at
        self.cooldown = cooldown
        self.log = log

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dispatcher, name)

    def execute(self, tool: str, arguments: Any, call_id: str) -> Any:
        self.wait()
        return self._dispatcher.execute(tool, arguments, call_id)

    def wait(self) -> None:
        started = time.monotonic()
        before = thermal_state()
        if self.cooldown:
            time.sleep(self.cooldown)
        state = thermal_state()
        while self.pause_at is not None and state >= self.pause_at:
            time.sleep(POLL_SECONDS)
            state = thermal_state()
        waited = time.monotonic() - started
        if waited > self.cooldown + 1:
            with self.log.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "thermal_state_before": before,
                            "thermal_state_after": state,
                            "paused_seconds": round(waited, 1),
                        }
                    )
                    + "\n"
                )
