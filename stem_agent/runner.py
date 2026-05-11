from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .backends import Backend
from .models import RunState, TurnResult, Usage
from .policies import BudgetPolicy, StopPolicy


@dataclass(frozen=True)
class TurnSnapshot:
    result: TurnResult
    session_id: str | None
    spent: Usage


@dataclass(frozen=True)
class AgentLoopOutcome:
    state: RunState
    snapshots: tuple[TurnSnapshot, ...]
    stop_reason: str | None = None
    error: str | None = None
    warnings: tuple[str, ...] = ()


class AgentLoop:
    def __init__(
        self,
        backend: Backend,
        budget_policy: BudgetPolicy,
        *,
        stop_policy: StopPolicy | None = None,
        state: RunState | None = None,
        allow_missing_usage: bool = False,
        events_log: str | Path | None = None,
    ) -> None:
        self.backend = backend
        self.budget_policy = budget_policy
        self.stop_policy = stop_policy or StopPolicy()
        self.state = state or RunState()
        self.allow_missing_usage = allow_missing_usage
        self.events_log = Path(events_log) if events_log else None
        self._events_log_initialized = False

    def run(self, initial_prompt: str) -> AgentLoopOutcome:
        prompt = initial_prompt
        snapshots: list[TurnSnapshot] = []
        warnings: list[str] = []

        while True:
            reason = self.budget_policy.stop_reason(self.state)
            if reason:
                return AgentLoopOutcome(
                    self.state,
                    tuple(snapshots),
                    stop_reason=reason,
                    warnings=tuple(warnings),
                )

            try:
                result = self.backend.run(prompt, self.state.session_id)
                self._write_events(result)
            except RuntimeError as error:
                return AgentLoopOutcome(
                    self.state,
                    tuple(snapshots),
                    error=str(error),
                    warnings=tuple(warnings),
                )
            except OSError as error:
                return AgentLoopOutcome(
                    self.state,
                    tuple(snapshots),
                    error=f"events_log_write_failed:{error}",
                    warnings=tuple(warnings),
                )

            if not result.saw_usage and not self.allow_missing_usage:
                return AgentLoopOutcome(
                    self.state,
                    tuple(snapshots),
                    error="missing_token_usage",
                    warnings=tuple(warnings),
                )

            if not result.saw_usage:
                warnings.append("missing_token_usage")

            self.state.record(result)
            snapshots.append(TurnSnapshot(result, self.state.session_id, self.state.spent))

            reason = self.stop_policy.stop_reason(result) or self.budget_policy.stop_reason(
                self.state
            )
            if reason:
                return AgentLoopOutcome(
                    self.state,
                    tuple(snapshots),
                    stop_reason=reason,
                    warnings=tuple(warnings),
                )

            prompt = self.continuation_prompt()

    def continuation_prompt(self) -> str:
        return build_continuation_prompt(
            self.state.spent,
            self.budget_policy.output_budget,
            self.budget_policy.total_budget,
        )

    def _write_events(self, result: TurnResult) -> None:
        if self.events_log is None:
            return

        mode = "a" if self._events_log_initialized else "w"
        with self.events_log.open(mode, encoding="utf-8") as handle:
            for event in result.raw_events:
                handle.write(json.dumps(event, separators=(",", ":")))
                handle.write("\n")
        self._events_log_initialized = True


def build_continuation_prompt(
    spent: Usage,
    output_budget: int,
    total_budget: int | None = None,
) -> str:
    remaining_output = max(output_budget - spent.output_tokens, 0)
    parts = [
        "Continue.",
        f"Output tokens spent: {spent.output_tokens}.",
        f"Remaining output token budget: {remaining_output}.",
    ]
    if total_budget is not None:
        remaining_total = max(total_budget - spent.total_tokens, 0)
        parts += [
            f"Total tokens spent: {spent.total_tokens}.",
            f"Remaining total token budget: {remaining_total}.",
        ]
    parts.append("Say STOP when no more calls are needed.")
    return " ".join(parts)
