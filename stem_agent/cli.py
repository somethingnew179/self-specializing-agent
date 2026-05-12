from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .backends import CodexExecBackend
from .debug_log import DebugLog
from .graph_runner import GraphRunner
from .models import RunState, Usage
from .policies import BudgetPolicy
from .runner import AgentLoop, TurnSnapshot


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        build_parser().print_help()
        return 0
    if argv[0] in {"-h", "--help"}:
        build_parser().parse_args(argv)
        return 0
    if argv and argv[0] == "graph":
        parser = build_graph_parser()
        args = parser.parse_args(argv[1:])
        validate_graph_args(parser, args)
        return run_graph_mode(args)
    if argv and argv[0] == "run":
        parser = build_run_parser()
        args = parser.parse_args(argv[1:])
        validate_run_args(parser, args)
        return run_direct_mode(parser, args)

    parser = build_legacy_parser()
    args = parser.parse_args(argv)
    validate_legacy_args(parser, args)

    if args.graph:
        return run_graph_mode(args)

    return run_direct_mode(parser, args)


def run_direct_mode(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    output_budget = resolve_output_budget(parser, args)
    debug_log = DebugLog(args.debug_log)
    prompt = read_prompt(args.prompt)
    debug_log.write(
        "cli_invocation",
        mode="run",
        args=vars(args),
        prompt=prompt,
    )
    backend = CodexExecBackend.from_args(args, debug_log=debug_log, debug_label="run")
    loop = AgentLoop(
        backend,
        BudgetPolicy(output_budget, args.total_budget, args.max_turns),
        state=RunState(session_id=args.resume),
        allow_missing_usage=args.allow_missing_usage,
        events_log=args.events_log,
    )
    outcome = loop.run(prompt)
    debug_log.write(
        "cli_finished",
        mode="run",
        error=outcome.error,
        stop_reason=outcome.stop_reason,
        snapshots=len(outcome.snapshots),
        warnings=outcome.warnings,
    )

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


def run_graph_mode(args: argparse.Namespace) -> int:
    project = Path(getattr(args, "project", ".")).expanduser().resolve()
    smart_defaults = getattr(args, "smart_defaults", False)
    graph_path = resolve_project_path(project, args.graph or ".agents/graph.json")
    events_log = None
    if args.events_log:
        events_log = resolve_project_path(project, args.events_log)
    elif smart_defaults:
        events_log = resolve_project_path(project, ".agents/run.jsonl")
    cd = resolve_project_path(project, args.cd) if args.cd else str(project)
    debug_log_path = (
        resolve_project_path(project, args.debug_log)
        if getattr(args, "debug_log", None)
        else None
    )
    debug_log = DebugLog(debug_log_path)
    prompt = read_prompt(args.prompt)
    debug_log.write(
        "cli_invocation",
        mode="graph",
        project=str(project),
        args=vars(args),
        prompt=prompt,
        graph_path=graph_path,
        events_log=events_log,
        debug_log=debug_log_path,
        cd=cd,
    )

    runner = GraphRunner(
        graph_path,
        events_log=events_log,
        model=args.model,
        cd=cd,
        sandbox=args.sandbox,
        skip_git_repo_check=args.skip_git_repo_check,
        allow_missing_usage=args.allow_missing_usage,
        max_steps=args.graph_max_steps,
        max_nodes=args.max_nodes,
        architect_retries=args.architect_retries,
        console_log=True,
        debug_log=debug_log,
        auto_review=args.auto_review,
    )
    outcome = runner.run(prompt)
    debug_log.write(
        "cli_finished",
        mode="graph",
        error=outcome.error,
        stop_reason=outcome.stop_reason,
        steps=len(outcome.context),
        usage=getattr(outcome, "usage", Usage()).__dict__,
    )

    if outcome.error:
        print(f"error={outcome.error}", file=sys.stderr)
        return 1

    if outcome.context:
        print(
            "result="
            + json.dumps(outcome.context[-1]["result"], separators=(",", ":"), sort_keys=True)
        )
    if outcome.stop_reason:
        print(f"stop={outcome.stop_reason} steps={len(outcome.context)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_graph_arguments(subparsers.add_parser("graph"))
    add_run_arguments(subparsers.add_parser("run"))
    return parser


def build_legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_run_arguments(parser)
    parser.add_argument("--graph")
    parser.add_argument("--project", default=".")
    parser.add_argument("--graph-max-steps", type=int, default=20)
    parser.add_argument("--max-nodes", type=int, default=8)
    parser.add_argument("--architect-retries", type=int, default=2)
    parser.set_defaults(smart_defaults=False)
    return parser


def build_graph_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stem-agent graph")
    add_graph_arguments(parser)
    parser.set_defaults(smart_defaults=True)
    return parser


def build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stem-agent run")
    add_run_arguments(parser)
    return parser


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("prompt")
    parser.add_argument("--budget", type=int, help="Deprecated alias for --output-budget")
    parser.add_argument("--output-budget", type=int)
    parser.add_argument("--total-budget", type=int)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--model")
    parser.add_argument("--cd")
    parser.add_argument("--sandbox")
    parser.add_argument("--skip-git-repo-check", action="store_true")
    parser.add_argument(
        "--auto-review",
        action="store_true",
        help="Use Codex auto-review approval mode for generated commands",
    )
    parser.add_argument(
        "--allow-missing-usage",
        action="store_true",
        help="Continue with zero token usage if codex does not emit usage",
    )
    parser.add_argument("--events-log")
    parser.add_argument("--debug-log", help="Write a verbose JSONL debug trace")


def add_graph_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("prompt")
    parser.add_argument("--project", default=".")
    parser.add_argument("--graph")
    parser.add_argument("--events-log")
    parser.add_argument("--debug-log", help="Write a verbose JSONL debug trace")
    parser.add_argument("--model")
    parser.add_argument("--cd")
    parser.add_argument("--sandbox", default="workspace-write")
    parser.add_argument(
        "--auto-review",
        action="store_true",
        help="Use Codex auto-review approval mode for generated commands",
    )
    parser.add_argument(
        "--skip-git-repo-check",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--graph-max-steps", type=int, default=20)
    parser.add_argument("--max-nodes", type=int, default=8)
    parser.add_argument("--architect-retries", type=int, default=2)
    parser.add_argument(
        "--allow-missing-usage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue with zero token usage if codex does not emit usage",
    )


def resolve_output_budget(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.output_budget is not None and args.budget is not None:
        parser.error("use either --output-budget or --budget, not both")

    output_budget = args.output_budget if args.output_budget is not None else args.budget
    if output_budget is None:
        parser.error("--output-budget is required")
    if output_budget < 0:
        parser.error("--output-budget must be >= 0")
    return output_budget


def validate_legacy_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.graph:
        validate_graph_args(parser, args)
    else:
        validate_run_args(parser, args)


def validate_run_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    resolve_output_budget(parser, args)
    if args.total_budget is not None and args.total_budget < 0:
        parser.error("--total-budget must be >= 0")
    if args.max_turns is not None and args.max_turns < 0:
        parser.error("--max-turns must be >= 0")
    if args.resume and args.cd:
        parser.error("--cd cannot be used with --resume")
    if args.resume and args.sandbox:
        parser.error("--sandbox cannot be used with --resume")


def validate_graph_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.graph_max_steps < 0:
        parser.error("--graph-max-steps must be >= 0")
    if args.max_nodes < 1:
        parser.error("--max-nodes must be >= 1")
    if args.architect_retries < 0:
        parser.error("--architect-retries must be >= 0")


def resolve_project_path(project: Path, value: str | Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project / path
    return str(path.resolve())


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
