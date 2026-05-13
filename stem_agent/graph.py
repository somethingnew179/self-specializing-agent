from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .json_schema import check_schema, validate_instance

ARCHITECT_NODE = "__architect__"
END_NODE = "__end__"
DEFAULT_MODEL = "gpt-5.5"

ARCHITECT_BOOTSTRAP_PROMPT = (
    "You are the Architect of a stem-agent graph. You do not execute the "
    "user's task. Design or repair the smallest useful graph, then choose "
    "the next worker node."
)


@dataclass(frozen=True)
class AgentSettings:
    model: str | None = None
    effort: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NodeOutput:
    route: str
    result: dict[str, Any]


def bootstrap_graph(model: str | None = None) -> dict[str, Any]:
    return {
        "version": 1,
        "start": "architect",
        "architect": {
            "model": model or DEFAULT_MODEL,
            "effort": "high",
            "prompt": ARCHITECT_BOOTSTRAP_PROMPT,
        },
        "nodes": {},
    }


def ensure_graph_file(path: str | Path, *, model: str | None = None) -> bool:
    graph_path = Path(path)
    if graph_path.exists() and graph_path.read_text(encoding="utf-8").strip():
        return False
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    write_graph(graph_path, bootstrap_graph(model))
    return True


def load_graph(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("graph root must be an object")
    return value


def write_graph(path: str | Path, graph: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(graph, handle, indent=2, sort_keys=True)
        handle.write("\n")


def validate_graph(graph: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if graph.get("version") != 1:
        errors.append("$.version: must be 1")

    start = graph.get("start")
    if not isinstance(start, str) or not start:
        errors.append("$.start: must be a non-empty string")

    architect = graph.get("architect")
    if not isinstance(architect, dict):
        errors.append("$.architect: must be an object")
    else:
        errors.extend(_validate_agent_settings("$.architect", architect, require_prompt=True))

    nodes = graph.get("nodes")
    if not isinstance(nodes, dict):
        errors.append("$.nodes: must be an object")
        nodes = {}

    if isinstance(start, str) and start not in {"architect", ARCHITECT_NODE, END_NODE}:
        if start not in nodes:
            errors.append(f"$.start: unknown node {start!r}")

    for node_id, node in nodes.items():
        if not isinstance(node_id, str) or not node_id:
            errors.append("$.nodes: node ids must be non-empty strings")
            continue
        if node_id in {ARCHITECT_NODE, END_NODE, "architect"}:
            errors.append(f"$.nodes.{node_id}: reserved node id")
        if not isinstance(node, dict):
            errors.append(f"$.nodes.{node_id}: must be an object")
            continue
        errors.extend(_validate_node(node_id, node, nodes))

    return errors


def parse_agent_settings(value: dict[str, Any]) -> AgentSettings:
    params = value.get("params", {})
    return AgentSettings(
        model=value.get("model"),
        effort=value.get("effort"),
        params=params if isinstance(params, dict) else {},
    )


def parse_node_output(text: str, result_schema: dict[str, Any]) -> tuple[NodeOutput | None, list[str]]:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as error:
        return None, [f"node output must be a JSON object: {error.msg}"]

    if not isinstance(value, dict):
        return None, ["node output must be a JSON object"]
    route = value.get("route")
    result = value.get("result")
    errors: list[str] = []
    if not isinstance(route, str) or not route:
        errors.append("$.route: must be a non-empty string")
    if not isinstance(result, dict):
        errors.append("$.result: must be an object")
    elif isinstance(result_schema, dict):
        errors.extend(validate_instance(result, result_schema))
    if errors:
        return None, errors
    return NodeOutput(route, result), []


def parse_architect_output(text: str) -> tuple[str | None, str | None, list[str]]:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as error:
        return None, None, [f"architect output must be JSON: {error.msg}"]
    if not isinstance(value, dict):
        return None, None, ["architect output must be a JSON object"]
    bug_report = value.get("bug_report")
    if bug_report is not None:
        if not isinstance(bug_report, str) or not bug_report.strip():
            return None, None, ["$.bug_report: must be a non-empty string"]
        return None, bug_report.strip(), []
    next_node = value.get("next_node")
    if not isinstance(next_node, str) or not next_node:
        return None, None, ["$.next_node: must be a non-empty string"]
    return next_node, None, []


def build_node_prompt(
    *,
    user_task: str,
    node_id: str,
    node: dict[str, Any],
    context: list[dict[str, Any]],
) -> str:
    allowed_routes = sorted(node["routes"].keys())
    payload = {
        "user_task": user_task,
        "node_id": node_id,
        "prompt": node["prompt"],
        "agent_settings": {
            "model": node.get("model"),
            "effort": node.get("effort"),
            "params": node.get("params", {}),
        },
        "result_schema": node["result_schema"],
        "allowed_routes": allowed_routes,
        "context": context,
        "response_format": {
            "route": "one of allowed_routes",
            "result": "object matching result_schema",
        },
    }
    return (
        "You are executing one node in a small agent graph. "
        "Do not describe the graph. Return only one JSON object.\n"
        + json.dumps(payload, indent=2, sort_keys=True)
    )


def build_architect_prompt(
    *,
    user_task: str,
    graph_path: Path,
    graph: dict[str, Any] | None,
    architect_prompt: str,
    context: list[dict[str, Any]],
    issue: str,
    errors: list[str],
    max_nodes: int,
) -> str:
    payload = {
        "user_task": user_task,
        "graph_path": str(graph_path),
        "max_nodes": max_nodes,
        "issue": issue,
        "errors": errors,
        "current_graph": graph,
        "context": context,
    }
    return "\n".join(
        [
            "ROLE",
            (
                "You are the Architect of a stem-agent graph. You do not execute "
                "the user's task. Your job is only to design or repair the graph "
                "and choose the next node."
            ),
            "",
            "HARD LIMITS",
            f"- You may only modify graph_path: {graph_path}.",
            "- You may create or edit minimal .agents metadata only if needed.",
            "- Do not implement the user's task yourself.",
            "- Do not create or edit product/source files for the user's task.",
            "- Do not run verification, build, install, browser, or test commands for the user's task.",
            "- If you are tempted to solve the task, create a worker node that solves it instead.",
            "",
            "LAST-RESORT BUG REPORT",
            "- If stem-agent, the sandbox, graph_path editing, or the graph contract behaves differently from these instructions and you cannot make progress safely, you may report a tool bug.",
            "- This is only for failures in your own architect workflow or the surrounding tool, not for normal task difficulty, uncertainty, graph validation errors, or worker failures.",
            "- Use this only as a last resort after graph repair or routing is not possible.",
            "- If you return a bug report, stem-agent will print it as an error and stop the run immediately.",
            "",
            "GRAPH DESIGN RULES",
            "- Prefer the smallest graph that can safely complete the task.",
            f"- Create up to {max_nodes} worker nodes.",
            "- Choose graph complexity based on task complexity.",
            "- For trivial tasks, one worker node is enough.",
            "- For small implementation tasks, prefer a short implement -> verify -> fix/done loop.",
            "- For research-heavy tasks, create research or planning nodes before implementation.",
            "- For ambiguous or multi-skill tasks, split work into specialized roles.",
            "- Do not over-engineer: use the smallest graph that can reliably complete the task.",
            "- Each node must have one clear responsibility.",
            f"- Use {ARCHITECT_NODE} when the graph needs repair or redesign.",
            f"- Use {END_NODE} only after a worker node has completed or verified the task.",
            "",
            "DECOMPOSITION STRATEGY",
            "- For non-trivial tasks, prefer many small worker runs over one long high-effort worker run.",
            "- Scope each worker to a small inspect, plan, implement, review, fix, or verify step.",
            "- Do not create one solve-all worker for a task with separable parts.",
            "- Do not default to a two-node solve -> validate graph when the task has uncertainty or multiple parts.",
            "- Prefer low or medium effort workers for narrow steps; reserve high effort for genuinely hard reasoning nodes.",
            f"- Add routes back to {ARCHITECT_NODE} when a worker may discover new constraints, blocked checks, ambiguity, or a need to redesign.",
            "- Use the graph loop to inspect partial results, intervene, split follow-up work, and get full value from the architecture.",
            "",
            "NEW TASK START MODE",
            "- Stem-agent calls you first for every user request, even when current_graph is valid and already has worker nodes.",
            "- When issue is new_task_start, compare user_task to current_graph before choosing next_node.",
            "- If current_graph fits the new user_task, leave graph_path unchanged and route to the best worker node.",
            "- If current_graph does not fit, rewrite graph_path with the smallest useful graph for the new user_task.",
            "- Assume a new user_task often needs a changed graph; do not blindly reuse an old structure.",
            "",
            "FINAL VALIDATION MODE",
            f"- When issue is final_validation, read context as the completed worker outputs.",
            f"- Return {{\"next_node\":\"{END_NODE}\"}} only if the user's task is complete and the final worker outputs are credible.",
            "- If the task is incomplete, verification is missing, or a worker admits an important check was blocked, route to the best worker node to continue.",
            "- You may repair graph_path first if no existing worker can handle the remaining work.",
            "- Do not finish just because the graph reached __end__; finish only when the context supports it.",
            "",
            "REQUIRED GRAPH SHAPE",
            "- The graph file must be one JSON object with: version, start, architect, nodes.",
            "- version must be 1.",
            "- start must be architect, __architect__, __end__, or an existing worker node id.",
            "- architect must be an object with prompt and optional model, effort, params.",
            "- nodes must be an object whose keys are worker node ids.",
            "- Every worker node must contain prompt, result_schema, and routes.",
            "- model, effort, and params are optional per worker node.",
            "- routes is a top-level node field. It is required and must be a non-empty object.",
            "- routes maps the route word returned by the worker to the next node id.",
            f"- Every routes target must be an existing worker node id, {ARCHITECT_NODE}, or {END_NODE}.",
            "- Do not put routes only in the worker prompt or only in result_schema.",
            "",
            "WORKER NODE TEMPLATE",
            "{",
            f"  \"model\": \"{DEFAULT_MODEL}\",",
            "  \"effort\": \"medium\",",
            "  \"prompt\": \"Tell the worker exactly what to do and which route words it may return.\",",
            "  \"result_schema\": {",
            "    \"type\": \"object\",",
            "    \"additionalProperties\": false,",
            "    \"properties\": {\"summary\": {\"type\": \"string\"}},",
            "    \"required\": [\"summary\"]",
            "  },",
            "  \"routes\": {\"done\": \"__end__\", \"needs_repair\": \"__architect__\"}",
            "}",
            "",
            "NODE DESIGN RULES",
            "- Delegate all implementation, inspection, fixing, and verification work to nodes.",
            "- You may create any worker roles that fit the task.",
            "- Common examples include programmer, designer, researcher, tester, reviewer, planner, critic, debugger, and documenter.",
            "- You may create subroles when useful, such as frontend_programmer, game_logic_programmer, ui_designer, accessibility_reviewer, test_writer, or bug_fixer.",
            "- You are not limited to these examples. Invent task-specific roles when they make the graph clearer or safer.",
            "- Node prompts must be operational and specific.",
            "- A node may edit files or run commands only when its prompt explicitly says so.",
            "- Each node must return only JSON with route and result.",
            "- Each node.result_schema must be valid JSON Schema for the inner result object only.",
            "- node.result_schema must not describe the whole worker response wrapper.",
            "- Do not put top-level response keys route or result inside node.result_schema unless the inner result object really has those fields.",
            "- The worker response route must be one of that node's top-level routes keys.",
            "- The worker response result must match that node's result_schema.",
            "- If a worker prompt lists allowed route words, the list must exactly match the node.routes keys.",
            "",
            "COMMON INVALID GRAPH TO AVOID",
            "- Invalid: result_schema contains route/result, but the node has no top-level routes.",
            "- Valid: result_schema describes only result fields, and top-level routes maps route words to next nodes.",
            "",
            "FINAL SELF-CHECK BEFORE RETURNING",
            "- Re-read the graph you wrote before returning.",
            "- Confirm every worker node has a non-empty top-level routes object.",
            "- Confirm every route target is defined or is a special target.",
            "- Confirm result_schema validates only the worker's result object.",
            "- Confirm your next_node is an existing worker node id or __end__.",
            "",
            "OUTPUT CONTRACT",
            "- Edit graph_path directly so it is a valid version 1 graph.",
            "- Normal success: Return only JSON in the form {\"next_node\":\"node_id\"}.",
            "- Last-resort tool bug: Return only JSON in the form {\"bug_report\":\"short actionable bug report\"}.",
            "- Do not include explanation, markdown, or any other text in your final response.",
            "",
            "ARCHITECT-SPECIFIC PROMPT FROM GRAPH",
            architect_prompt or "(none)",
            "",
            "INPUT",
            json.dumps(payload, indent=2, sort_keys=True),
        ]
    )


def _validate_node(
    node_id: str,
    node: dict[str, Any],
    nodes: dict[str, Any],
) -> list[str]:
    path = f"$.nodes.{node_id}"
    errors = _validate_agent_settings(path, node, require_prompt=True)

    result_schema = node.get("result_schema")
    if not isinstance(result_schema, dict):
        errors.append(f"{path}.result_schema: must be an object")
    else:
        errors.extend(check_schema(result_schema, f"{path}.result_schema"))

    routes = node.get("routes")
    if not isinstance(routes, dict) or not routes:
        errors.append(f"{path}.routes: must be a non-empty object")
    else:
        for route, target in routes.items():
            if not isinstance(route, str) or not route:
                errors.append(f"{path}.routes: route names must be non-empty strings")
            if not isinstance(target, str) or not target:
                errors.append(f"{path}.routes.{route}: target must be a non-empty string")
            elif target not in {ARCHITECT_NODE, END_NODE} and target not in nodes:
                errors.append(f"{path}.routes.{route}: unknown target {target!r}")

    return errors


def _validate_agent_settings(
    path: str,
    value: dict[str, Any],
    *,
    require_prompt: bool,
) -> list[str]:
    errors: list[str] = []
    if require_prompt and not isinstance(value.get("prompt"), str):
        errors.append(f"{path}.prompt: must be a string")
    if "model" in value and value["model"] is not None and not isinstance(value["model"], str):
        errors.append(f"{path}.model: must be a string")
    if "effort" in value and value["effort"] is not None and not isinstance(value["effort"], str):
        errors.append(f"{path}.effort: must be a string")
    if "params" in value and value["params"] is not None and not isinstance(value["params"], dict):
        errors.append(f"{path}.params: must be an object")
    return errors
