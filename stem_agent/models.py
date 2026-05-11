from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "Usage":
        input_tokens = _int_token(value, "input_tokens")
        output_tokens = _int_token(value, "output_tokens")
        return cls(
            input_tokens=input_tokens,
            cached_input_tokens=_int_token(value, "cached_input_tokens"),
            output_tokens=output_tokens,
            reasoning_output_tokens=_int_token(value, "reasoning_output_tokens"),
            total_tokens=_int_token(value, "total_tokens", input_tokens + output_tokens),
        )

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_output_tokens=(
                self.reasoning_output_tokens + other.reasoning_output_tokens
            ),
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass(frozen=True)
class TurnResult:
    session_id: str | None
    last_text: str
    usage: Usage
    saw_usage: bool
    raw_events: tuple[dict[str, Any], ...] = ()


@dataclass
class RunState:
    session_id: str | None = None
    spent: Usage = field(default_factory=Usage)
    turns: list[TurnResult] = field(default_factory=list)

    def record(self, result: TurnResult) -> None:
        self.session_id = result.session_id or self.session_id
        self.spent = self.spent + result.usage
        self.turns.append(result)

    @property
    def raw_events(self) -> tuple[dict[str, Any], ...]:
        events: list[dict[str, Any]] = []
        for turn in self.turns:
            events.extend(turn.raw_events)
        return tuple(events)


def _int_token(
    value: Mapping[str, Any] | None,
    key: str,
    default: int = 0,
) -> int:
    return int((value or {}).get(key, default) or 0)
