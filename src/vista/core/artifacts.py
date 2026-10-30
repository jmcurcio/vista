"""Host-only, allowlisted attempt records and immutable display evidence."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any

from .contracts import ObservationBundle
from .json import canonical


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.root.chmod(0o700)
        self._lock = threading.Lock()

    def manifest(self, value: dict[str, Any]) -> None:
        with (self.root / "manifest.json").open("x", encoding="utf-8") as stream:
            stream.write(canonical(value) + "\n")
        (self.root / "manifest.json").chmod(0o600)

    def record(self, event: str, value: dict[str, Any]) -> None:
        with self._lock:
            path = self.root / "events.jsonl"
            with path.open("a", encoding="utf-8") as stream:
                stream.write(canonical({"event": event, **value}) + "\n")
            path.chmod(0o600)

    def evidence(self, bundle: ObservationBundle) -> None:
        for observation in bundle.observations:
            for visual in observation.views:
                digest = hashlib.sha256(visual.data).hexdigest()
                path = self.root / "evidence" / digest
                # Created with the first image, so a store that archives
                # none (the task-level record) leaves no empty directory.
                path.parent.mkdir(mode=0o700, exist_ok=True)
                try:
                    with path.open("xb") as stream:
                        stream.write(visual.data)
                    path.chmod(0o400)
                except FileExistsError:
                    if path.read_bytes() != visual.data:
                        raise RuntimeError(
                            "Archived evidence has been modified"
                        ) from None
