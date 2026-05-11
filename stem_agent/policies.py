from __future__ import annotations

from dataclasses import dataclass

from .models import RunState, TurnResult

STOP_MARKER = "STOP"


@dataclass(frozen=True)
class BudgetPolicy:
    output_budget: int
    total_budget: int | None = None
    max_turns: int | None = None

    def stop_reason(self, state: RunState) -> str | None:
        if self.max_turns is not None and len(state.turns) >= self.max_turns:
            return "max_turns_spent"
        if state.spent.output_tokens >= self.output_budget:
            return "output_budget_spent"
        if (
            self.total_budget is not None
            and state.spent.total_tokens >= self.total_budget
        ):
            return "total_budget_spent"
        return None


@dataclass(frozen=True)
class StopPolicy:
    marker: str = STOP_MARKER

    def stop_reason(self, result: TurnResult) -> str | None:
        if self.marker and self.marker in result.last_text:
            return "model_stop"
        return None
