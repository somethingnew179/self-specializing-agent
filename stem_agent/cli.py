from __future__ import annotations

import argparse
import sys

from .backends import CodexExecBackend
from .models import RunState
from .policies import BudgetPolicy
from .runner import AgentLoop, TurnSnapshot


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_budget = resolve_output_budget(parser, args)
    validate_args(parser, args)

    backend = CodexExecBackend.from_args(args)
    loop = AgentLoop(
        backend,
        BudgetPolicy(output_budget, args.total_budget, args.max_turns),
        state=RunState(session_id=args.resume),
        allow_missing_usage=args.allow_missing_usage,
        events_log=args.events_log,
    )
    outcome = loop.run(read_prompt(args.prompt))

    for warning in outcome.warnings:
        print(f"warning={warning}", file=sys.stderr)

    for snapshot in outcome.snapshots:
        print_turn(snapshot, output_budget, args.total_budget)

    if outcome.error:
        print(f"error={outcome.error}", file=sys.stderr)
        return 1

    if outcome.stop_reason:
        print(f"stop={outcome.stop_reason}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--budget", type=int, help="Deprecated alias for --output-budget")
    parser.add_argument("--output-budget", type=int)
    parser.add_argument("--total-budget", type=int)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--model")
    parser.add_argument("--cd")
    parser.add_argument("--sandbox")
    parser.add_argument(
        "--allow-missing-usage",
        action="store_true",
        help="Continue with zero token usage if codex does not emit usage",
    )
    parser.add_argument("--events-log")
    return parser


def resolve_output_budget(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.output_budget is not None and args.budget is not None:
        parser.error("use either --output-budget or --budget, not both")

    output_budget = args.output_budget if args.output_budget is not None else args.budget
    if output_budget is None:
        parser.error("--output-budget is required")
    if output_budget < 0:
        parser.error("--output-budget must be >= 0")
    return output_budget


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.total_budget is not None and args.total_budget < 0:
        parser.error("--total-budget must be >= 0")
    if args.max_turns is not None and args.max_turns < 0:
        parser.error("--max-turns must be >= 0")
    if args.resume and args.cd:
        parser.error("--cd cannot be used with --resume")
    if args.resume and args.sandbox:
        parser.error("--sandbox cannot be used with --resume")


def read_prompt(value: str) -> str:
    if value == "-":
        return sys.stdin.read()
    return value


def print_turn(
    snapshot: TurnSnapshot,
    output_budget: int,
    total_budget: int | None,
) -> None:
    result = snapshot.result
    remaining_output = max(output_budget - snapshot.spent.output_tokens, 0)
    fields = [
        f"session={snapshot.session_id or '-'}",
        f"turn_input_tokens={result.usage.input_tokens}",
        f"turn_cached_input_tokens={result.usage.cached_input_tokens}",
        f"turn_output_tokens={result.usage.output_tokens}",
        f"turn_reasoning_output_tokens={result.usage.reasoning_output_tokens}",
        f"turn_total_tokens={result.usage.total_tokens}",
        f"spent_output={snapshot.spent.output_tokens}",
        f"remaining_output={remaining_output}",
    ]
    if total_budget is not None:
        remaining_total = max(total_budget - snapshot.spent.total_tokens, 0)
        fields += [
            f"spent_total={snapshot.spent.total_tokens}",
            f"remaining_total={remaining_total}",
        ]
    print(" ".join(fields))
