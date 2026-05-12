from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


class JsonlEventLog:
    def __init__(
        self,
        path: str | Path | None,
        *,
        echo: bool = False,
        stream: TextIO | None = None,
    ) -> None:
        self.path = Path(path) if path else None
        self.echo = echo
        self.stream = stream or sys.stderr
        self._initialized = False

    def write(self, event_type: str, **fields: Any) -> None:
        event = {
            "type": event_type,
            "time": datetime.now(timezone.utc).isoformat(),
            **fields,
        }

        if self.path is not None:
            mode = "a" if self._initialized else "w"
            with self.path.open(mode, encoding="utf-8") as handle:
                handle.write(json.dumps(event, separators=(",", ":"), sort_keys=True))
                handle.write("\n")
            self._initialized = True

        if self.echo:
            message = format_human_event(event_type, fields)
            if message:
                print(message, file=self.stream, flush=True)


def format_human_event(event_type: str, fields: dict[str, Any]) -> str | None:
    if event_type == "graph_created":
        return f"[graph] created {fields.get('graph_path', '-')}"
    if event_type == "graph_validation_failed":
        errors = fields.get("errors") or []
        return f"[graph] validation failed: {len(errors)} error(s)"
    if event_type == "architect_called":
        session = fields.get("session_id")
        resume = f" session={session}" if session else ""
        return (
            "[agent] architect start"
            f" issue={fields.get('issue', '-')}"
            f" attempt={fields.get('attempt', 0)}"
            f" model={fields.get('model') or '-'}"
            f" effort={fields.get('effort') or '-'}"
            f"{resume}"
        )
    if event_type == "architect_finished":
        return f"[agent] architect done next={fields.get('next_node', '-')}"
    if event_type == "architect_failed":
        error = fields.get("error")
        errors = fields.get("errors")
        detail = error or (errors[0] if errors else "-")
        return f"[agent] architect failed: {detail}"
    if event_type == "architect_bug_report":
        return f"[agent] architect bug: {fields.get('bug_report', '-')}"
    if event_type == "retry":
        return (
            f"[runner] retry {fields.get('target', '-')}"
            f" attempt={fields.get('attempt', '-')}"
        )
    if event_type == "node_called":
        routes = ",".join(fields.get("routes") or [])
        return (
            f"[agent] node start {fields.get('node', '-')}"
            f" model={fields.get('model') or '-'}"
            f" effort={fields.get('effort') or '-'}"
            f" routes={routes or '-'}"
        )
    if event_type == "node_result":
        return (
            f"[agent] node done {fields.get('node', '-')}"
            f" route={fields.get('route', '-')}"
        )
    if event_type == "node_result_invalid":
        errors = fields.get("errors") or []
        detail = errors[0] if errors else "-"
        return f"[agent] node invalid {fields.get('node', '-')}: {detail}"
    if event_type == "node_failed":
        return f"[agent] node failed {fields.get('node', '-')}: {fields.get('error', '-')}"
    if event_type == "transition":
        return (
            f"[runner] transition {fields.get('node', '-')}"
            f" --{fields.get('route', '-')}--> {fields.get('next_node', '-')}"
        )
    if event_type == "final_validation_reopened":
        return f"[runner] final validation routed to {fields.get('next_node', '-')}"
    if event_type == "run_finished":
        return (
            f"[runner] finished {fields.get('stop_reason', '-')}"
            f" steps={fields.get('steps', '-')}"
        )
    return None
