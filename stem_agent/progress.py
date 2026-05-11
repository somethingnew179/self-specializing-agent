from __future__ import annotations

import sys
import threading
import time
from typing import TextIO

from .models import TurnResult


class SingleLineProgress:
    def __init__(
        self,
        label: str,
        *,
        stream: TextIO | None = None,
        interval: float = 0.5,
    ) -> None:
        self.label = label
        self.stream = stream or sys.stderr
        self.interval = interval
        self.started_at = 0.0
        self.event_count = 0
        self.last_event = "-"
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            self.started_at = time.monotonic()
            self._running = True
            self._write_running_locked()
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()

    def event(self, event_type: str) -> None:
        with self._lock:
            self.event_count += 1
            self.last_event = event_type or "-"
            self._write_running_locked()

    def finish(self, result: TurnResult) -> None:
        self._stop_thread()
        elapsed = self._elapsed()
        if result.saw_usage:
            usage = result.usage
            suffix = f"output={usage.output_tokens} total={usage.total_tokens}"
        else:
            suffix = "usage=missing"
        self._write_final(
            f"[{self.label}] done {elapsed}s events={self.event_count} {suffix}"
        )

    def failed(self, message: str) -> None:
        self._stop_thread()
        self._write_final(f"[{self.label}] failed {self._elapsed()}s error={message}")

    def _heartbeat(self) -> None:
        while True:
            time.sleep(self.interval)
            with self._lock:
                if not self._running:
                    return
                self._write_running_locked()

    def _stop_thread(self) -> None:
        with self._lock:
            self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 0.1)

    def _write_running_locked(self) -> None:
        self._write_line(
            f"[{self.label}] running {self._elapsed()}s "
            f"events={self.event_count} last={self.last_event}"
        )

    def _write_final(self, message: str) -> None:
        self._write_line(message)
        print(file=self.stream, flush=True)

    def _write_line(self, message: str) -> None:
        print(f"\r\033[2K{message}", end="", file=self.stream, flush=True)

    def _elapsed(self) -> int:
        if not self.started_at:
            return 0
        return int(time.monotonic() - self.started_at)
