from __future__ import annotations

import json
from typing import Any

from .models import TurnResult, Usage

AUTO_REVIEW_CONFIG_OVERRIDE = 'approvals_reviewer="auto_review"'


def parse_usage(value: dict[str, Any] | None) -> Usage:
    return Usage.from_dict(value)


def parse_events(stdout: str) -> TurnResult:
    accumulator = CodexEventAccumulator()

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue

        accumulator.add(parse_event_line(line))

    return accumulator.result()


def parse_event_line(line: str) -> dict[str, Any]:
    event = json.loads(line)
    if not isinstance(event, dict):
        raise ValueError("codex event must be a JSON object")
    return event


class CodexEventAccumulator:
    def __init__(self) -> None:
        self.session_id: str | None = None
        self.last_text = ""
        self.usage = Usage()
        self.saw_usage = False
        self.raw_events: list[dict[str, Any]] = []

    def add(self, event: dict[str, Any]) -> None:
        self.raw_events.append(event)
        event_type = event.get("type")

        if event_type == "thread.started":
            self.session_id = event.get("thread_id")
        elif event_type == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                self.last_text = item.get("text", "")
        elif event_type == "turn.completed":
            event_usage = event.get("usage")
            if event_usage:
                self.usage = parse_usage(event_usage)
                self.saw_usage = True

    def result(self) -> TurnResult:
        return TurnResult(
            self.session_id,
            self.last_text,
            self.usage,
            self.saw_usage,
            tuple(self.raw_events),
        )


def build_codex_command(
    prompt: str,
    session_id: str | None = None,
    *,
    model: str | None = None,
    cd: str | None = None,
    sandbox: str | None = None,
    skip_git_repo_check: bool = False,
    auto_review: bool = False,
    config_overrides: list[str] | tuple[str, ...] = (),
) -> list[str]:
    config_overrides = _effective_config_overrides(config_overrides, auto_review)
    if session_id:
        command = ["codex", "exec", "resume", "--json"]
        for override in config_overrides:
            command += ["-c", override]
        if model:
            command += ["--model", model]
        if skip_git_repo_check:
            command.append("--skip-git-repo-check")
        command += [session_id, prompt]
        return command

    command = ["codex", "exec", "--json"]
    for override in config_overrides:
        command += ["-c", override]
    if model:
        command += ["--model", model]
    if cd:
        command += ["--cd", cd]
    if sandbox:
        command += ["--sandbox", sandbox]
    if skip_git_repo_check:
        command.append("--skip-git-repo-check")
    command.append(prompt)
    return command


def _effective_config_overrides(
    config_overrides: list[str] | tuple[str, ...],
    auto_review: bool,
) -> tuple[str, ...]:
    overrides = tuple(config_overrides)
    if not auto_review or AUTO_REVIEW_CONFIG_OVERRIDE in overrides:
        return overrides
    return (AUTO_REVIEW_CONFIG_OVERRIDE, *overrides)
