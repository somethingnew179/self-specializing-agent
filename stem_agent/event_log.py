from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonlEventLog:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._initialized = False

    def write(self, event_type: str, **fields: Any) -> None:
        if self.path is None:
            return

        event = {
            "type": event_type,
            "time": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        mode = "a" if self._initialized else "w"
        with self.path.open(mode, encoding="utf-8") as handle:
            handle.write(json.dumps(event, separators=(",", ":"), sort_keys=True))
            handle.write("\n")
        self._initialized = True
