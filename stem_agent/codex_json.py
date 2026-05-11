from __future__ import annotations

import json
from typing import Any

from .models import TurnResult, Usage


def parse_usage(value: dict[str, Any] | None) -> Usage:
    return Usage.from_dict(value)


def parse_events(stdout: str) -> TurnResult:
    session_id = None
    last_text = ""
    usage = Usage()
    saw_usage = False
    raw_events: list[dict[str, Any]] = []

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue

        event = json.loads(line)
        raw_events.append(event)
        event_type = event.get("type")

        if event_type == "thread.started":
            session_id = event.get("thread_id")
        elif event_type == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                last_text = item.get("text", "")
        elif event_type == "turn.completed":
            event_usage = event.get("usage")
            if event_usage:
                usage = parse_usage(event_usage)
                saw_usage = True

    return TurnResult(session_id, last_text, usage, saw_usage, tuple(raw_events))


def build_codex_command(
    prompt: str,
    session_id: str | None = None,
    *,
    model: str | None = None,
    cd: str | None = None,
    sandbox: str | None = None,
) -> list[str]:
    if session_id:
        command = ["codex", "exec", "resume", "--json"]
        if model:
            command += ["--model", model]
        command += [session_id, prompt]
        return command

    command = ["codex", "exec", "--json"]
    if model:
        command += ["--model", model]
    if cd:
        command += ["--cd", cd]
    if sandbox:
        command += ["--sandbox", sandbox]
    command.append(prompt)
    return command
