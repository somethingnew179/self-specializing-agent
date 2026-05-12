from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DebugLog:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._initialized = False

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def write(self, event_type: str, **fields: Any) -> None:
        if self.path is None:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "type": event_type,
            "time": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        mode = "a" if self._initialized else "w"
        with self.path.open(mode, encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    event,
                    default=str,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            handle.write("\n")
        self._initialized = True
