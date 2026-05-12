from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .backends import Backend, CodexExecBackend
from .event_log import JsonlEventLog
from .graph import (
    ARCHITECT_NODE,
    END_NODE,
    AgentSettings,
    build_architect_prompt,
    build_node_prompt,
    ensure_graph_file,
    load_graph,
    parse_agent_settings,
    parse_architect_output,
    parse_node_output,
    validate_graph,
)
from .models import TurnResult, Usage
from .progress import SingleLineProgress

BackendFactory = Callable[[AgentSettings], Backend]


@dataclass(frozen=True)
class GraphRunOutcome:
    stop_reason: str | None = None
    error: str | None = None
    context: list[dict] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


class GraphRunner:
    def __init__(
        self,
        graph_path: str | Path,
        *,
        backend_factory: BackendFactory | None = None,
        events_log: str | Path | None = None,
        model: str | None = None,
        cd: str | None = None,
        sandbox: str | None = None,
        skip_git_repo_check: bool = False,
        allow_missing_usage: bool = False,
        max_steps: int = 20,
        max_nodes: int = 8,
        architect_retries: int = 2,
        console_log: bool = False,
    ) -> None:
        self.graph_path = Path(graph_path).resolve()
        self.events = JsonlEventLog(events_log)
        self.model = model
        self.cd = cd
        self.sandbox = sandbox
        self.skip_git_repo_check = skip_git_repo_check
        self.allow_missing_usage = allow_missing_usage
        self.max_steps = max_steps
        self.max_nodes = max_nodes
        self.architect_retries = architect_retries
        self.console_log = console_log
        self.backend_factory = backend_factory

    def run(self, user_task: str) -> GraphRunOutcome:
        context: list[dict] = []
        usage = Usage()
        graph_created = ensure_graph_file(self.graph_path, model=self.model)
        if graph_created:
            self.events.write("graph_created", graph_path=str(self.graph_path))

        graph, errors = self._load_and_validate_graph()
        if errors:
            next_node, graph, architect_usage, error = self._call_architect_until_valid(
                user_task,
                context,
                graph,
                "graph_validation_failed",
                errors,
            )
            usage = usage + architect_usage
            if error:
                return GraphRunOutcome(error=error, context=context, usage=usage)
            current = next_node
        else:
            current = graph.get("start", "architect")

        if current in {"architect", ARCHITECT_NODE} or not graph.get("nodes"):
            next_node, graph, architect_usage, error = self._call_architect_until_valid(
                user_task,
                context,
                graph,
                "bootstrap_or_architect_start",
                [],
            )
            usage = usage + architect_usage
            if error:
                return GraphRunOutcome(error=error, context=context, usage=usage)
            current = next_node

        steps = 0
        while True:
            if current == END_NODE:
                self.events.write("run_finished", stop_reason="graph_finished", steps=steps)
                return GraphRunOutcome("graph_finished", context=context, usage=usage)
            if steps >= self.max_steps:
                self.events.write("run_finished", stop_reason="max_steps_spent", steps=steps)
                return GraphRunOutcome("max_steps_spent", context=context, usage=usage)
            if current in {"architect", ARCHITECT_NODE}:
                next_node, graph, architect_usage, error = self._call_architect_until_valid(
                    user_task,
                    context,
                    graph,
                    "architect_route",
                    [],
                )
                usage = usage + architect_usage
                if error:
                    return GraphRunOutcome(error=error, context=context, usage=usage)
                current = next_node
                continue

            graph, errors = self._load_and_validate_graph()
            if errors:
                next_node, graph, architect_usage, error = self._call_architect_until_valid(
                    user_task,
                    context,
                    graph,
                    "graph_validation_failed",
                    errors,
                )
                usage = usage + architect_usage
                if error:
                    return GraphRunOutcome(error=error, context=context, usage=usage)
                current = next_node
                continue

            nodes = graph["nodes"]
            node = nodes.get(current)
            if not isinstance(node, dict):
                next_node, graph, architect_usage, error = self._call_architect_until_valid(
                    user_task,
                    context,
                    graph,
                    "missing_node",
                    [f"missing node {current!r}"],
                )
                usage = usage + architect_usage
                if error:
                    return GraphRunOutcome(error=error, context=context, usage=usage)
                current = next_node
                continue

            result, step_usage, error = self._run_node(user_task, current, node, context)
            usage = usage + step_usage
            if error:
                next_node, graph, architect_usage, architect_error = self._call_architect_until_valid(
                    user_task,
                    context,
                    graph,
                    "node_failed",
                    [error],
                )
                usage = usage + architect_usage
                if architect_error:
                    return GraphRunOutcome(error=architect_error, context=context, usage=usage)
                current = next_node
                continue

            assert result is not None
            target = node["routes"][result["route"]]
            context.append(
                {
                    "node": current,
                    "route": result["route"],
                    "result": result["result"],
                }
            )
            self.events.write(
                "transition",
                node=current,
                route=result["route"],
                next_node=target,
            )
            current = target
            steps += 1

    def _backend_for(self, settings: AgentSettings, label: str) -> Backend:
        if self.backend_factory is not None:
            return self.backend_factory(settings)
        return self._default_backend_factory(settings, label)

    def _default_backend_factory(
        self,
        settings: AgentSettings,
        label: str = "",
    ) -> Backend:
        return CodexExecBackend(
            model=settings.model or self.model,
            cd=self.cd,
            skip_git_repo_check=self.skip_git_repo_check,
            config_overrides=self._codex_permission_overrides(),
            progress=SingleLineProgress(label) if self.console_log else None,
        )

    def _codex_permission_overrides(self) -> tuple[str, str]:
        project_root = self._project_root_for_permissions()
        agents_dir = self.graph_path.parent
        filesystem = ",".join(
            [
                f"{json.dumps(str(project_root))}=\"write\"",
                f"{json.dumps(str(agents_dir))}=\"write\"",
            ]
        )
        return (
            'default_permissions="stem-agent-write"',
            f"permissions.stem-agent-write={{filesystem={{{filesystem}}}}}",
        )

    def _project_root_for_permissions(self) -> Path:
        if self.cd:
            return Path(self.cd).expanduser().resolve()
        if self.graph_path.parent.name == ".agents":
            return self.graph_path.parent.parent
        return self.graph_path.parent

    def _load_and_validate_graph(self) -> tuple[dict, list[str]]:
        try:
            graph = load_graph(self.graph_path)
        except (OSError, ValueError) as error:
            return {}, [str(error)]
        errors = validate_graph(graph)
        if errors:
            self.events.write(
                "graph_validation_failed",
                graph_path=str(self.graph_path),
                errors=errors,
            )
        return graph, errors

    def _run_node(
        self,
        user_task: str,
        node_id: str,
        node: dict,
        context: list[dict],
    ) -> tuple[dict | None, Usage, str | None]:
        settings = parse_agent_settings(node)
        prompt = build_node_prompt(
            user_task=user_task,
            node_id=node_id,
            node=node,
            context=context,
        )
        self.events.write(
            "node_called",
            node=node_id,
            model=settings.model or self.model,
            effort=settings.effort,
            params=settings.params,
            routes=sorted(node["routes"].keys()),
        )
        try:
            turn = self._backend_for(settings, f"node:{node_id}").run(prompt)
        except RuntimeError as error:
            self.events.write("node_failed", node=node_id, error=str(error))
            return None, Usage(), str(error)

        missing_usage = self._missing_usage_error(turn)
        if missing_usage:
            self.events.write("node_failed", node=node_id, error=missing_usage)
            return None, turn.usage, missing_usage

        parsed, errors = parse_node_output(turn.last_text, node["result_schema"])
        if errors or parsed is None:
            self.events.write(
                "node_result_invalid",
                node=node_id,
                errors=errors,
                raw_text=turn.last_text,
                usage=turn.usage.__dict__,
            )
            return None, turn.usage, "; ".join(errors)

        if parsed.route not in node["routes"]:
            error = f"unknown route {parsed.route!r}"
            self.events.write("node_result_invalid", node=node_id, errors=[error])
            return None, turn.usage, error

        result = {"route": parsed.route, "result": parsed.result}
        self.events.write(
            "node_result",
            node=node_id,
            route=parsed.route,
            result=parsed.result,
            usage=turn.usage.__dict__,
        )
        return result, turn.usage, None

    def _call_architect_until_valid(
        self,
        user_task: str,
        context: list[dict],
        graph: dict,
        issue: str,
        errors: list[str],
    ) -> tuple[str, dict, Usage, str | None]:
        usage = Usage()
        current_errors = errors
        current_issue = issue
        current_graph = graph

        for attempt in range(self.architect_retries + 1):
            next_node, current_graph, attempt_usage, error = self._call_architect(
                user_task,
                context,
                current_graph,
                current_issue,
                current_errors,
                attempt,
            )
            usage = usage + attempt_usage
            if error:
                if error.startswith("architect_bug_report:"):
                    return "", current_graph, usage, error
                current_issue = "architect_failed"
                current_errors = [error]
                continue

            loaded_graph, graph_errors = self._load_and_validate_graph()
            if graph_errors:
                current_graph = loaded_graph
                current_issue = "graph_validation_failed"
                current_errors = graph_errors
                if attempt < self.architect_retries:
                    self.events.write("retry", target="architect", attempt=attempt + 1)
                continue

            if next_node not in loaded_graph["nodes"] and next_node != END_NODE:
                current_graph = loaded_graph
                current_issue = "architect_next_node_invalid"
                current_errors = [f"unknown next_node {next_node!r}"]
                if attempt < self.architect_retries:
                    self.events.write("retry", target="architect", attempt=attempt + 1)
                continue

            return next_node, loaded_graph, usage, None

        return "", current_graph, usage, "graph_validation_error"

    def _call_architect(
        self,
        user_task: str,
        context: list[dict],
        graph: dict,
        issue: str,
        errors: list[str],
        attempt: int,
    ) -> tuple[str, dict, Usage, str | None]:
        architect = graph.get("architect", {}) if isinstance(graph, dict) else {}
        settings = parse_agent_settings(architect if isinstance(architect, dict) else {})
        prompt = build_architect_prompt(
            user_task=user_task,
            graph_path=self.graph_path,
            graph=graph,
            architect_prompt=(
                architect.get("prompt", "")
                if isinstance(architect, dict)
                else ""
            ),
            context=context,
            issue=issue,
            errors=errors,
            max_nodes=self.max_nodes,
        )
        self.events.write(
            "architect_called",
            issue=issue,
            attempt=attempt,
            model=settings.model or self.model,
            effort=settings.effort,
            params=settings.params,
            errors=errors,
        )
        try:
            turn = self._backend_for(settings, "architect").run(prompt)
        except RuntimeError as error:
            self.events.write("architect_failed", attempt=attempt, error=str(error))
            return "", graph, Usage(), str(error)

        missing_usage = self._missing_usage_error(turn)
        if missing_usage:
            self.events.write("architect_failed", attempt=attempt, error=missing_usage)
            return "", graph, turn.usage, missing_usage

        next_node, bug_report, output_errors = parse_architect_output(turn.last_text)
        if bug_report:
            error = f"architect_bug_report:{bug_report}"
            self.events.write(
                "architect_bug_report",
                attempt=attempt,
                bug_report=bug_report,
                usage=turn.usage.__dict__,
            )
            return "", graph, turn.usage, error

        if output_errors or next_node is None:
            self.events.write(
                "architect_failed",
                attempt=attempt,
                errors=output_errors,
                raw_text=turn.last_text,
                usage=turn.usage.__dict__,
            )
            return "", graph, turn.usage, "; ".join(output_errors)

        self.events.write(
            "architect_finished",
            attempt=attempt,
            next_node=next_node,
            usage=turn.usage.__dict__,
        )
        return next_node, graph, turn.usage, None

    def _missing_usage_error(self, turn: TurnResult) -> str | None:
        if turn.saw_usage or self.allow_missing_usage:
            return None
        return "missing_token_usage"
