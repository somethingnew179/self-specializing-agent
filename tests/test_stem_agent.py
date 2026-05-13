import io
import json
import os
import tomllib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stem_agent import RunState, TurnResult, Usage
from stem_agent.backends import CodexExecBackend
from stem_agent.cli import main
from stem_agent.codex_json import build_codex_command, parse_events, parse_usage
from stem_agent.debug_log import DebugLog
from stem_agent.graph import (
    END_NODE,
    AgentSettings,
    bootstrap_graph,
    build_architect_prompt,
    build_node_prompt,
    parse_architect_output,
    parse_node_output,
    validate_graph,
    write_graph,
)
from stem_agent.graph_runner import GraphRunner
from stem_agent.policies import BudgetPolicy, StopPolicy
from stem_agent.progress import SingleLineProgress


def graph_turn(text, output_tokens=1, saw_usage=True):
    return TurnResult(
        "session",
        text,
        Usage(output_tokens=output_tokens, total_tokens=output_tokens),
        saw_usage,
    )


def final_validation_turn():
    return graph_turn('{"next_node":"__end__"}')


class FakeBackend:
    def __init__(self, actions):
        self.actions = list(actions)
        self.prompts = []
        self.sessions = []

    def run(self, prompt, session_id=None):
        self.prompts.append(prompt)
        self.sessions.append(session_id)
        if not self.actions:
            raise RuntimeError("no fake backend actions left")
        action = self.actions.pop(0)
        if callable(action):
            return action(prompt)
        return action


class FakePopen:
    def __init__(self, lines, returncode=0, stderr=""):
        self.stdout = iter(lines)
        self.stderr = io.StringIO(stderr)
        self.returncode = returncode
        self.killed = False

    def wait(self):
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


class ProgressFakeBackend:
    def __init__(self, *args, **kwargs):
        self.progress = kwargs.get("progress")
        self.debug_label = kwargs.get("debug_label")

    def run(self, prompt, session_id=None):
        if self.debug_label == "architect":
            result = graph_turn('{"next_node":"__end__"}')
        else:
            result = graph_turn('{"route":"done","result":{"summary":"ok"}}')
        if self.progress:
            self.progress.start()
            self.progress.event("thread.started")
            self.progress.finish(result)
        return result


class FakeGraphRunner:
    calls = []

    def __init__(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def run(self, prompt):
        self.prompt = prompt
        return type(
            "Outcome",
            (),
            {
                "error": None,
                "context": [{"result": {"summary": "ok"}}],
                "stop_reason": "graph_finished",
            },
        )()


class StemAgentTests(unittest.TestCase):
    def test_parse_events_with_full_usage(self):
        stdout = "\n".join(
            [
                '{"type":"thread.started","thread_id":"abc"}',
                '{"type":"item.completed","item":{"type":"agent_message","text":"work"}}',
                (
                    '{"type":"turn.completed","usage":'
                    '{"input_tokens":99,"cached_input_tokens":80,'
                    '"output_tokens":7,"reasoning_output_tokens":3,'
                    '"total_tokens":106}}'
                ),
            ]
        )

        result = parse_events(stdout)

        self.assertEqual(result.session_id, "abc")
        self.assertEqual(result.last_text, "work")
        self.assertTrue(result.saw_usage)
        self.assertEqual(result.usage.input_tokens, 99)
        self.assertEqual(result.usage.cached_input_tokens, 80)
        self.assertEqual(result.usage.output_tokens, 7)
        self.assertEqual(result.usage.reasoning_output_tokens, 3)
        self.assertEqual(result.usage.total_tokens, 106)
        self.assertEqual(len(result.raw_events), 3)

    def test_missing_usage_is_zero(self):
        stdout = '{"type":"item.completed","item":{"type":"agent_message","text":"work"}}'

        result = parse_events(stdout)

        self.assertIsNone(result.session_id)
        self.assertEqual(result.last_text, "work")
        self.assertEqual(result.usage, Usage())
        self.assertFalse(result.saw_usage)

    def test_usage_total_defaults_to_input_plus_output(self):
        usage = parse_usage({"input_tokens": 5, "output_tokens": 7})

        self.assertEqual(usage.total_tokens, 12)

    def test_stop_marker_stops(self):
        result = TurnResult(None, "done STOP", Usage(output_tokens=1), True)

        self.assertEqual(StopPolicy().stop_reason(result), "model_stop")

    def test_output_budget_stops(self):
        state = RunState()
        state.record(TurnResult(None, "continue", Usage(output_tokens=10), True))

        self.assertEqual(
            BudgetPolicy(output_budget=10).stop_reason(state),
            "output_budget_spent",
        )

    def test_total_budget_stops(self):
        state = RunState()
        state.record(
            TurnResult(
                None,
                "continue",
                Usage(input_tokens=9, output_tokens=1, total_tokens=10),
                True,
            )
        )

        self.assertEqual(
            BudgetPolicy(output_budget=100, total_budget=10).stop_reason(state),
            "total_budget_spent",
        )

    def test_max_turns_stops(self):
        state = RunState()
        state.record(TurnResult(None, "continue", Usage(), True))

        self.assertEqual(
            BudgetPolicy(output_budget=100, max_turns=1).stop_reason(state),
            "max_turns_spent",
        )
        self.assertEqual(state.raw_events, ())

    def test_build_fresh_codex_command(self):
        self.assertEqual(
            build_codex_command(
                "hello",
                model="gpt-5.5",
                cd="/tmp/x",
                sandbox="read-only",
            ),
            [
                "codex",
                "exec",
                "--json",
                "--model",
                "gpt-5.5",
                "--cd",
                "/tmp/x",
                "--sandbox",
                "read-only",
                "hello",
            ],
        )

    def test_build_fresh_codex_command_can_skip_git_repo_check(self):
        self.assertEqual(
            build_codex_command(
                "hello",
                cd="/tmp/x",
                skip_git_repo_check=True,
            ),
            [
                "codex",
                "exec",
                "--json",
                "--cd",
                "/tmp/x",
                "--skip-git-repo-check",
                "hello",
            ],
        )

    def test_build_fresh_codex_command_can_add_config_overrides(self):
        self.assertEqual(
            build_codex_command(
                "hello",
                cd="/tmp/x",
                config_overrides=(
                    'default_permissions="stem-agent-write"',
                    'permissions.stem-agent-write={filesystem={"/tmp/x"="write","/tmp/x/.agents"="write"}}',
                ),
                skip_git_repo_check=True,
            ),
            [
                "codex",
                "exec",
                "--json",
                "-c",
                'default_permissions="stem-agent-write"',
                "-c",
                'permissions.stem-agent-write={filesystem={"/tmp/x"="write","/tmp/x/.agents"="write"}}',
                "--cd",
                "/tmp/x",
                "--skip-git-repo-check",
                "hello",
            ],
        )

    def test_build_fresh_codex_command_can_enable_auto_review(self):
        self.assertEqual(
            build_codex_command(
                "hello",
                cd="/tmp/x",
                auto_review=True,
            ),
            [
                "codex",
                "exec",
                "--json",
                "-c",
                'approvals_reviewer="auto_review"',
                "--cd",
                "/tmp/x",
                "hello",
            ],
        )

    def test_build_resume_codex_command_keeps_options_before_session(self):
        self.assertEqual(
            build_codex_command(
                "hello",
                "session-1",
                model="gpt-5.5",
                cd="/tmp/x",
                sandbox="read-only",
                config_overrides=('default_permissions="stem-agent-write"',),
                skip_git_repo_check=True,
            ),
            [
                "codex",
                "exec",
                "resume",
                "--json",
                "-c",
                'default_permissions="stem-agent-write"',
                "--model",
                "gpt-5.5",
                "--skip-git-repo-check",
                "session-1",
                "hello",
            ],
        )

    def test_build_resume_codex_command_can_enable_auto_review(self):
        self.assertEqual(
            build_codex_command(
                "hello",
                "session-1",
                model="gpt-5.5",
                auto_review=True,
                config_overrides=('default_permissions="stem-agent-write"',),
                skip_git_repo_check=True,
            ),
            [
                "codex",
                "exec",
                "resume",
                "--json",
                "-c",
                'approvals_reviewer="auto_review"',
                "-c",
                'default_permissions="stem-agent-write"',
                "--model",
                "gpt-5.5",
                "--skip-git-repo-check",
                "session-1",
                "hello",
            ],
        )

    def test_streaming_codex_backend_reports_progress_and_usage(self):
        lines = [
            '{"type":"thread.started","thread_id":"abc"}\n',
            '{"type":"item.completed","item":{"type":"agent_message","text":"STOP"}}\n',
            '{"type":"turn.completed","usage":{"input_tokens":4,"output_tokens":3,"total_tokens":7}}\n',
        ]
        stderr = io.StringIO()
        progress = SingleLineProgress("architect", stream=stderr, interval=0.01)

        with patch(
            "stem_agent.backends.subprocess.Popen",
            return_value=FakePopen(lines),
        ):
            result = CodexExecBackend(progress=progress).run("hello")

        self.assertEqual(result.session_id, "abc")
        self.assertEqual(result.last_text, "STOP")
        self.assertEqual(result.usage.output_tokens, 3)
        self.assertIn("[architect] running", stderr.getvalue())
        self.assertIn("[architect] done", stderr.getvalue())
        self.assertIn("output=3 total=7", stderr.getvalue())

    def test_streaming_codex_backend_reports_missing_usage(self):
        lines = [
            '{"type":"thread.started","thread_id":"abc"}\n',
            '{"type":"item.completed","item":{"type":"agent_message","text":"STOP"}}\n',
        ]
        stderr = io.StringIO()
        progress = SingleLineProgress("node:work", stream=stderr, interval=0.01)

        with patch(
            "stem_agent.backends.subprocess.Popen",
            return_value=FakePopen(lines),
        ):
            result = CodexExecBackend(progress=progress).run("hello")

        self.assertFalse(result.saw_usage)
        self.assertIn("[node:work] done", stderr.getvalue())
        self.assertIn("usage=missing", stderr.getvalue())

    def test_streaming_codex_backend_runs_from_configured_cd(self):
        lines = [
            '{"type":"thread.started","thread_id":"abc"}\n',
            '{"type":"item.completed","item":{"type":"agent_message","text":"STOP"}}\n',
        ]
        progress = SingleLineProgress("architect", stream=io.StringIO(), interval=0.01)

        with patch(
            "stem_agent.backends.subprocess.Popen",
            return_value=FakePopen(lines),
        ) as popen:
            result = CodexExecBackend(
                cd="/tmp/x",
                skip_git_repo_check=True,
                progress=progress,
            ).run("hello", "session-1")

        self.assertEqual(result.session_id, "abc")
        self.assertEqual(popen.call_args.kwargs["cwd"], "/tmp/x")
        self.assertIn("--skip-git-repo-check", popen.call_args.args[0])

    def test_streaming_codex_backend_writes_debug_log(self):
        lines = [
            '{"type":"thread.started","thread_id":"abc"}\n',
            '{"type":"item.completed","item":{"type":"agent_message","text":"STOP"}}\n',
            '{"type":"turn.completed","usage":{"input_tokens":4,"output_tokens":3,"total_tokens":7}}\n',
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            debug_path = os.path.join(tmpdir, "debug.jsonl")
            progress = SingleLineProgress(
                "architect",
                stream=io.StringIO(),
                interval=0.01,
            )

            with patch(
                "stem_agent.backends.subprocess.Popen",
                return_value=FakePopen(lines, stderr="warn"),
            ):
                result = CodexExecBackend(
                    progress=progress,
                    debug_log=DebugLog(debug_path),
                    debug_label="architect",
                ).run("hello")

            self.assertEqual(result.last_text, "STOP")
            with open(debug_path, encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle]

        self.assertEqual(events[0]["type"], "codex_exec_started")
        self.assertEqual(events[0]["label"], "architect")
        self.assertEqual(events[0]["prompt"], "hello")
        self.assertIn("codex_stdout_line", [event["type"] for event in events])
        self.assertIn("codex_event", [event["type"] for event in events])
        self.assertIn("codex_result_parsed", [event["type"] for event in events])
        self.assertEqual(events[-2]["stderr"], "warn")
        self.assertEqual(events[-1]["usage"]["total_tokens"], 7)

    def test_zero_budget_does_not_call_codex(self):
        with patch("stem_agent.backends.CodexExecBackend.run") as backend_run:
            with patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(main(["hello", "--budget", "0"]), 0)

        backend_run.assert_not_called()

    def test_codex_error_returns_failure(self):
        with patch(
            "stem_agent.backends.CodexExecBackend.run", side_effect=RuntimeError("boom")
        ):
            with patch("sys.stderr", new=io.StringIO()):
                self.assertEqual(main(["hello", "--budget", "10"]), 1)

    def test_missing_usage_returns_failure_by_default(self):
        result = TurnResult(None, "work", Usage(), False)

        with patch("stem_agent.backends.CodexExecBackend.run", return_value=result):
            with patch("sys.stderr", new=io.StringIO()) as stderr:
                self.assertEqual(main(["hello", "--budget", "10"]), 1)

        self.assertIn("error=missing_token_usage", stderr.getvalue())

    def test_allow_missing_usage_keeps_zero_usage_behavior(self):
        first = TurnResult(None, "work", Usage(), False)
        second = TurnResult(
            "abc",
            "STOP",
            Usage(output_tokens=1, total_tokens=1),
            True,
        )

        with patch("stem_agent.backends.CodexExecBackend.run", side_effect=[first, second]):
            with patch("sys.stdout", new=io.StringIO()) as stdout:
                with patch("sys.stderr", new=io.StringIO()) as stderr:
                    self.assertEqual(
                        main(["hello", "--budget", "10", "--allow-missing-usage"]),
                        0,
                    )

        self.assertIn("warning=missing_token_usage", stderr.getvalue())
        self.assertIn("stop=model_stop", stdout.getvalue())

    def test_output_budget_flag_is_supported(self):
        with patch("stem_agent.backends.CodexExecBackend.run"):
            with patch("sys.stdout", new=io.StringIO()) as stdout:
                self.assertEqual(main(["hello", "--output-budget", "0"]), 0)

        self.assertIn("stop=output_budget_spent", stdout.getvalue())

    def test_run_subcommand_passes_auto_review_to_backend(self):
        backend = FakeBackend([graph_turn("STOP")])

        with patch(
            "stem_agent.cli.CodexExecBackend.from_args",
            return_value=backend,
        ) as from_args:
            with patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(
                    main(["run", "--auto-review", "--output-budget", "10", "STOP"]),
                    0,
                )

        args = from_args.call_args.args[0]
        self.assertTrue(args.auto_review)

    def test_legacy_run_mode_passes_auto_review_to_backend(self):
        backend = FakeBackend([graph_turn("STOP")])

        with patch(
            "stem_agent.cli.CodexExecBackend.from_args",
            return_value=backend,
        ) as from_args:
            with patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(
                    main(["--auto-review", "--output-budget", "10", "STOP"]),
                    0,
                )

        args = from_args.call_args.args[0]
        self.assertTrue(args.auto_review)

    def test_total_budget_stop_is_reported(self):
        result = TurnResult(
            "abc",
            "continue",
            Usage(input_tokens=7, output_tokens=3, total_tokens=10),
            True,
        )

        with patch("stem_agent.backends.CodexExecBackend.run", return_value=result):
            with patch("sys.stdout", new=io.StringIO()) as stdout:
                self.assertEqual(
                    main(["hello", "--output-budget", "100", "--total-budget", "10"]),
                    0,
                )

        self.assertIn("spent_total=10", stdout.getvalue())
        self.assertIn("stop=total_budget_spent", stdout.getvalue())

    def test_max_turns_stop_is_reported(self):
        result = TurnResult(
            "abc",
            "continue",
            Usage(output_tokens=1, total_tokens=1),
            True,
        )

        with patch("stem_agent.backends.CodexExecBackend.run", return_value=result):
            with patch("sys.stdout", new=io.StringIO()) as stdout:
                self.assertEqual(
                    main(["hello", "--output-budget", "100", "--max-turns", "1"]),
                    0,
                )

        self.assertIn("stop=max_turns_spent", stdout.getvalue())

    def test_output_budget_and_legacy_budget_conflict(self):
        with patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["hello", "--budget", "1", "--output-budget", "1"])

    def test_resume_rejects_cd(self):
        with patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["hello", "--budget", "1", "--resume", "abc", "--cd", "/tmp"])

    def test_resume_rejects_sandbox(self):
        with patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["hello", "--budget", "1", "--resume", "abc", "--sandbox", "read-only"])

    def test_events_log_writes_raw_jsonl(self):
        result = TurnResult(
            "abc",
            "STOP",
            Usage(output_tokens=1, total_tokens=1),
            True,
            raw_events=({"type": "thread.started", "thread_id": "abc"},),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "events.jsonl")
            with patch("stem_agent.backends.CodexExecBackend.run", return_value=result):
                with patch("sys.stdout", new=io.StringIO()):
                    self.assertEqual(
                        main(["hello", "--budget", "10", "--events-log", path]),
                        0,
                    )

            with open(path, encoding="utf-8") as handle:
                self.assertEqual(
                    handle.read().strip(),
                    '{"type":"thread.started","thread_id":"abc"}',
                )

    def test_cli_main_is_importable(self):
        with patch("stem_agent.backends.CodexExecBackend.run"):
            with patch("sys.stdout", new=io.StringIO()) as stdout:
                self.assertEqual(main(["hello", "--output-budget", "0"]), 0)

        self.assertIn("stop=output_budget_spent", stdout.getvalue())


class GraphTests(unittest.TestCase):
    def valid_graph(self):
        return {
            "version": 1,
            "start": "work",
            "architect": {
                "model": "gpt-5.5",
                "effort": "high",
                "prompt": "Fix the graph.",
            },
            "nodes": {
                "work": {
                    "model": "gpt-5.5",
                    "effort": "medium",
                    "prompt": "Do the work.",
                    "result_schema": {
                        "type": "object",
                        "required": ["summary"],
                        "properties": {"summary": {"type": "string"}},
                        "additionalProperties": False,
                    },
                    "routes": {"done": END_NODE},
                }
            },
        }

    def test_valid_graph_passes_validation(self):
        self.assertEqual(validate_graph(self.valid_graph()), [])

    def test_bootstrap_graph_has_strict_architect_prompt(self):
        graph = bootstrap_graph("gpt-test")

        prompt = graph["architect"]["prompt"]
        self.assertIn("You do not execute the user's task", prompt)
        self.assertIn("choose the next worker node", prompt)

    def test_architect_prompt_forbids_solving_task_directly(self):
        prompt = build_architect_prompt(
            user_task="сделай браузерный тетрис",
            graph_path=Path("/tmp/project/.agents/graph.json"),
            graph=self.valid_graph(),
            architect_prompt="old weak prompt",
            context=[],
            issue="bootstrap",
            errors=[],
            max_nodes=7,
        )

        self.assertIn("You do not execute the user's task", prompt)
        self.assertIn("You may only modify graph_path", prompt)
        self.assertIn("Do not implement the user's task yourself", prompt)
        self.assertIn("Do not create or edit product/source files", prompt)
        self.assertIn("Delegate all implementation", prompt)
        self.assertIn("Create up to 7 worker nodes", prompt)
        self.assertNotIn("usually 1-4 nodes", prompt)
        self.assertIn("Choose graph complexity based on task complexity", prompt)
        self.assertIn("FINAL VALIDATION MODE", prompt)
        self.assertIn("When issue is final_validation", prompt)
        self.assertIn("programmer, designer, researcher, tester", prompt)
        self.assertIn("frontend_programmer", prompt)
        self.assertIn("Invent task-specific roles", prompt)
        self.assertIn("LAST-RESORT BUG REPORT", prompt)
        self.assertIn('{"bug_report":"short actionable bug report"}', prompt)
        self.assertIn("stem-agent will print it as an error and stop", prompt)
        self.assertIn('Return only JSON in the form {"next_node":"node_id"}', prompt)

    def test_parse_architect_output_accepts_bug_report(self):
        next_node, bug_report, errors = parse_architect_output(
            '{"bug_report":"graph_path is not writable"}'
        )

        self.assertIsNone(next_node)
        self.assertEqual(bug_report, "graph_path is not writable")
        self.assertEqual(errors, [])

    def test_node_prompt_still_contains_execution_contract(self):
        graph = self.valid_graph()

        prompt = build_node_prompt(
            user_task="сделай браузерный тетрис",
            node_id="work",
            node=graph["nodes"]["work"],
            context=[{"node": "inspect", "route": "done", "result": {"summary": "ok"}}],
        )

        self.assertIn('"user_task":', prompt)
        self.assertIn('"allowed_routes":', prompt)
        self.assertIn('"result_schema":', prompt)
        self.assertIn('"context":', prompt)
        self.assertIn("Return only one JSON object", prompt)

    def test_graph_validation_rejects_unknown_route_target(self):
        graph = self.valid_graph()
        graph["nodes"]["work"]["routes"]["missing"] = "missing"

        errors = validate_graph(graph)

        self.assertTrue(any("unknown target 'missing'" in error for error in errors))

    def test_graph_validation_rejects_bad_result_schema(self):
        graph = self.valid_graph()
        graph["nodes"]["work"]["result_schema"] = {"type": "bad"}

        errors = validate_graph(graph)

        self.assertTrue(any("unsupported JSON Schema type" in error for error in errors))

    def test_parse_node_output_validates_result_schema(self):
        schema = {
            "type": "object",
            "required": ["summary"],
            "properties": {"summary": {"type": "string"}},
        }

        parsed, errors = parse_node_output(
            '{"route":"done","result":{"summary":"ok"}}',
            schema,
        )

        self.assertEqual(errors, [])
        self.assertEqual(parsed.route, "done")
        self.assertEqual(parsed.result, {"summary": "ok"})

    def test_parse_node_output_rejects_bad_result(self):
        schema = {
            "type": "object",
            "required": ["summary"],
            "properties": {"summary": {"type": "string"}},
        }

        parsed, errors = parse_node_output('{"route":"done","result":{}}', schema)

        self.assertIsNone(parsed)
        self.assertTrue(any("missing required property" in error for error in errors))

    def test_graph_runner_executes_node_and_logs_transition(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = os.path.join(tmpdir, "graph.json")
            log_path = os.path.join(tmpdir, "events.jsonl")
            write_graph(graph_path, self.valid_graph())
            backend = FakeBackend(
                [
                    graph_turn('{"route":"done","result":{"summary":"ok"}}'),
                    final_validation_turn(),
                ]
            )

            outcome = GraphRunner(
                graph_path,
                backend_factory=lambda settings: backend,
                events_log=log_path,
            ).run("task")

            self.assertEqual(outcome.stop_reason, "graph_finished")
            self.assertIsNone(outcome.error)
            self.assertEqual(outcome.context[-1]["result"], {"summary": "ok"})
            with open(log_path, encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle]
            self.assertIn("node_called", [event["type"] for event in events])
            self.assertIn("node_result", [event["type"] for event in events])
            self.assertIn("transition", [event["type"] for event in events])

    def test_graph_runner_resumes_architect_session_across_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = os.path.join(tmpdir, "graph.json")
            graph = self.valid_graph()
            graph["start"] = "architect"
            write_graph(graph_path, graph)
            first_backend = FakeBackend(
                [
                    TurnResult(
                        "architect-session-1",
                        '{"next_node":"work"}',
                        Usage(output_tokens=1, total_tokens=1),
                        True,
                    ),
                    graph_turn('{"route":"done","result":{"summary":"ok"}}'),
                    TurnResult(
                        "architect-session-1",
                        '{"next_node":"__end__"}',
                        Usage(output_tokens=1, total_tokens=1),
                        True,
                    ),
                ]
            )

            first = GraphRunner(
                graph_path,
                backend_factory=lambda settings: first_backend,
                allow_missing_usage=False,
            ).run("task")

            self.assertEqual(first.stop_reason, "graph_finished")
            session_path = Path(tmpdir) / "architect_session.json"
            with session_path.open(encoding="utf-8") as handle:
                self.assertEqual(
                    json.load(handle),
                    {"session_id": "architect-session-1"},
                )

            second_backend = FakeBackend(
                [
                    TurnResult(
                        "architect-session-1",
                        '{"next_node":"work"}',
                        Usage(output_tokens=1, total_tokens=1),
                        True,
                    ),
                    graph_turn('{"route":"done","result":{"summary":"again"}}'),
                    TurnResult(
                        "architect-session-1",
                        '{"next_node":"__end__"}',
                        Usage(output_tokens=1, total_tokens=1),
                        True,
                    ),
                ]
            )

            second = GraphRunner(
                graph_path,
                backend_factory=lambda settings: second_backend,
                allow_missing_usage=False,
            ).run("task")

            self.assertEqual(second.stop_reason, "graph_finished")
            self.assertEqual(
                second_backend.sessions,
                ["architect-session-1", None, "architect-session-1"],
            )

    def test_graph_runner_calls_architect_on_invalid_node_output(self):
        graph = self.valid_graph()
        graph["nodes"]["finish"] = {
            "prompt": "Finish.",
            "result_schema": {
                "type": "object",
                "required": ["summary"],
                "properties": {"summary": {"type": "string"}},
            },
            "routes": {"done": END_NODE},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = os.path.join(tmpdir, "graph.json")
            write_graph(graph_path, graph)
            backend = FakeBackend(
                [
                    graph_turn('{"route":"unknown","result":{"summary":"bad route"}}'),
                    graph_turn('{"next_node":"finish"}'),
                    graph_turn('{"route":"done","result":{"summary":"fixed"}}'),
                    final_validation_turn(),
                ]
            )

            outcome = GraphRunner(
                graph_path,
                backend_factory=lambda settings: backend,
            ).run("task")

            self.assertEqual(outcome.stop_reason, "graph_finished")
            self.assertEqual(outcome.context[-1]["node"], "finish")
            self.assertTrue(any("node_failed" in prompt for prompt in backend.prompts))

    def test_graph_runner_final_validation_can_continue_graph(self):
        graph = self.valid_graph()
        graph["nodes"]["fix"] = {
            "prompt": "Fix final issue.",
            "result_schema": {
                "type": "object",
                "required": ["summary"],
                "properties": {"summary": {"type": "string"}},
            },
            "routes": {"done": END_NODE},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = os.path.join(tmpdir, "graph.json")
            log_path = os.path.join(tmpdir, "events.jsonl")
            write_graph(graph_path, graph)
            backend = FakeBackend(
                [
                    graph_turn('{"route":"done","result":{"summary":"first"}}'),
                    graph_turn('{"next_node":"fix"}'),
                    graph_turn('{"route":"done","result":{"summary":"fixed"}}'),
                    final_validation_turn(),
                ]
            )

            outcome = GraphRunner(
                graph_path,
                backend_factory=lambda settings: backend,
                events_log=log_path,
            ).run("task")

            self.assertEqual(outcome.stop_reason, "graph_finished")
            self.assertEqual([item["node"] for item in outcome.context], ["work", "fix"])
            self.assertTrue(any("final_validation" in prompt for prompt in backend.prompts))
            with open(log_path, encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle]
            self.assertIn("final_validation_reopened", [event["type"] for event in events])

    def test_graph_runner_retries_architect_until_graph_is_valid(self):
        invalid_graph = self.valid_graph()
        invalid_graph["start"] = "missing"

        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = os.path.join(tmpdir, "graph.json")
            log_path = os.path.join(tmpdir, "events.jsonl")
            write_graph(graph_path, invalid_graph)

            def fix_graph(prompt):
                write_graph(graph_path, self.valid_graph())
                return graph_turn('{"next_node":"work"}')

            backend = FakeBackend(
                [
                    graph_turn('{"next_node":"work"}'),
                    fix_graph,
                    graph_turn('{"route":"done","result":{"summary":"ok"}}'),
                    final_validation_turn(),
                ]
            )

            outcome = GraphRunner(
                graph_path,
                backend_factory=lambda settings: backend,
                events_log=log_path,
                architect_retries=2,
            ).run("task")

            self.assertEqual(outcome.stop_reason, "graph_finished")
            with open(log_path, encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle]
            self.assertIn("retry", [event["type"] for event in events])

    def test_graph_runner_fails_closed_after_architect_retry_limit(self):
        invalid_graph = self.valid_graph()
        invalid_graph["start"] = "missing"

        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = os.path.join(tmpdir, "graph.json")
            write_graph(graph_path, invalid_graph)
            backend = FakeBackend([graph_turn('{"next_node":"work"}')])

            outcome = GraphRunner(
                graph_path,
                backend_factory=lambda settings: backend,
                architect_retries=0,
            ).run("task")

            self.assertEqual(outcome.error, "graph_validation_error")

    def test_graph_runner_stops_on_architect_bug_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = os.path.join(tmpdir, "graph.json")
            log_path = os.path.join(tmpdir, "events.jsonl")
            write_graph(graph_path, bootstrap_graph())
            backend = FakeBackend(
                [
                    graph_turn(
                        '{"bug_report":"graph_path is not writable"}'
                    )
                ]
            )

            outcome = GraphRunner(
                graph_path,
                backend_factory=lambda settings: backend,
                events_log=log_path,
                architect_retries=2,
            ).run("task")

            self.assertEqual(
                outcome.error,
                "architect_bug_report:graph_path is not writable",
            )
            self.assertEqual(backend.actions, [])
            with open(log_path, encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle]
            self.assertIn("architect_bug_report", [event["type"] for event in events])
            self.assertNotIn("retry", [event["type"] for event in events])

    def test_graph_runner_uses_permission_profile_for_agents_dir(self):
        project = Path("/tmp/project").resolve()
        graph_path = project / ".agents" / "graph.json"
        runner = GraphRunner(
            graph_path,
            cd=str(project),
            sandbox="workspace-write",
            skip_git_repo_check=True,
        )

        backend = runner._default_backend_factory(AgentSettings(), "architect")

        self.assertEqual(
            backend.build_codex_command("hello"),
            [
                "codex",
                "exec",
                "--json",
                "-c",
                'default_permissions="stem-agent-write"',
                "-c",
                (
                    'permissions.stem-agent-write={filesystem={'
                    '":root"="read",'
                    '":project_roots"="write",'
                    '":tmpdir"="write",'
                    '"/tmp"="write",'
                    f'{json.dumps(str(project))}="write",'
                    f'{json.dumps(str(graph_path.resolve().parent))}="write"'
                    "}}"
                ),
                "--cd",
                str(project),
                "--skip-git-repo-check",
                "hello",
            ],
        )

    def test_graph_runner_can_enable_auto_review(self):
        project = Path("/tmp/project").resolve()
        graph_path = project / ".agents" / "graph.json"
        runner = GraphRunner(
            graph_path,
            cd=str(project),
            skip_git_repo_check=True,
            auto_review=True,
        )

        backend = runner._default_backend_factory(AgentSettings(), "architect")

        command = backend.build_codex_command("hello", "session-1")
        self.assertIn("-c", command)
        self.assertIn('approvals_reviewer="auto_review"', command)
        self.assertIn("--skip-git-repo-check", command)

    def test_graph_cli_does_not_require_output_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = os.path.join(tmpdir, "graph.json")
            write_graph(graph_path, self.valid_graph())
            fake_backend = FakeBackend(
                [
                    graph_turn('{"route":"done","result":{"summary":"ok"}}'),
                    final_validation_turn(),
                ]
            )

            with patch("stem_agent.graph_runner.CodexExecBackend", return_value=fake_backend):
                with patch("sys.stdout", new=io.StringIO()) as stdout:
                    with patch("sys.stderr", new=io.StringIO()):
                        self.assertEqual(main(["--graph", graph_path, "task"]), 0)

            self.assertIn('result={"summary":"ok"}', stdout.getvalue())
            self.assertIn("stop=graph_finished", stdout.getvalue())

    def test_graph_cli_prints_basic_progress_to_stderr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = os.path.join(tmpdir, "graph.json")
            write_graph(graph_path, self.valid_graph())

            with patch("stem_agent.graph_runner.CodexExecBackend", ProgressFakeBackend):
                with patch("sys.stdout", new=io.StringIO()):
                    with patch("sys.stderr", new=io.StringIO()) as stderr:
                        self.assertEqual(main(["--graph", graph_path, "task"]), 0)

            self.assertIn("[node:work] running", stderr.getvalue())
            self.assertIn("[node:work] done", stderr.getvalue())

    def test_graph_subcommand_uses_project_relative_defaults(self):
        FakeGraphRunner.calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("stem_agent.cli.GraphRunner", FakeGraphRunner):
                with patch("sys.stdout", new=io.StringIO()):
                    with patch("sys.stderr", new=io.StringIO()):
                        self.assertEqual(
                            main(["graph", "--project", tmpdir, "task"]),
                            0,
                        )

            args, kwargs = FakeGraphRunner.calls[0]
            project = Path(tmpdir).resolve()
            self.assertEqual(args[0], str(project / ".agents" / "graph.json"))
            self.assertEqual(kwargs["events_log"], str(project / ".agents" / "run.jsonl"))
            self.assertEqual(kwargs["cd"], str(project))
            self.assertEqual(kwargs["sandbox"], "workspace-write")
            self.assertTrue(kwargs["skip_git_repo_check"])
            self.assertTrue(kwargs["allow_missing_usage"])
            self.assertEqual(kwargs["max_nodes"], 8)
            self.assertFalse(kwargs["auto_review"])

    def test_graph_subcommand_can_enable_auto_review(self):
        FakeGraphRunner.calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("stem_agent.cli.GraphRunner", FakeGraphRunner):
                with patch("sys.stdout", new=io.StringIO()):
                    with patch("sys.stderr", new=io.StringIO()):
                        self.assertEqual(
                            main(["graph", "--project", tmpdir, "--auto-review", "task"]),
                            0,
                        )

            _, kwargs = FakeGraphRunner.calls[0]
            self.assertTrue(kwargs["auto_review"])

    def test_legacy_graph_mode_can_enable_auto_review(self):
        FakeGraphRunner.calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("stem_agent.cli.GraphRunner", FakeGraphRunner):
                with patch("sys.stdout", new=io.StringIO()):
                    with patch("sys.stderr", new=io.StringIO()):
                        self.assertEqual(
                            main(
                                [
                                    "--project",
                                    tmpdir,
                                    "--graph",
                                    ".agents/graph.json",
                                    "--auto-review",
                                    "task",
                                ]
                            ),
                            0,
                        )

            _, kwargs = FakeGraphRunner.calls[0]
            self.assertTrue(kwargs["auto_review"])

    def test_graph_subcommand_resolves_custom_paths_under_project(self):
        FakeGraphRunner.calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("stem_agent.cli.GraphRunner", FakeGraphRunner):
                with patch("sys.stdout", new=io.StringIO()):
                    with patch("sys.stderr", new=io.StringIO()):
                        self.assertEqual(
                            main(
                                [
                                    "graph",
                                    "--project",
                                    tmpdir,
                                    "--graph",
                                    "custom/graph.json",
                                    "--events-log",
                                    "logs/run.jsonl",
                                    "--cd",
                                    "workspace",
                                    "--max-nodes",
                                    "12",
                                    "task",
                                ]
                            ),
                            0,
                        )

            args, kwargs = FakeGraphRunner.calls[0]
            project = Path(tmpdir).resolve()
            self.assertEqual(args[0], str(project / "custom" / "graph.json"))
            self.assertEqual(kwargs["events_log"], str(project / "logs" / "run.jsonl"))
            self.assertEqual(kwargs["cd"], str(project / "workspace"))
            self.assertEqual(kwargs["max_nodes"], 12)

    def test_graph_subcommand_resolves_debug_log_under_project(self):
        FakeGraphRunner.calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("stem_agent.cli.GraphRunner", FakeGraphRunner):
                with patch("sys.stdout", new=io.StringIO()):
                    with patch("sys.stderr", new=io.StringIO()):
                        self.assertEqual(
                            main(
                                [
                                    "graph",
                                    "--project",
                                    tmpdir,
                                    "--debug-log",
                                    "logs/debug.jsonl",
                                    "task",
                                ]
                            ),
                            0,
                        )

            _, kwargs = FakeGraphRunner.calls[0]
            project = Path(tmpdir).resolve()
            self.assertEqual(
                kwargs["debug_log"].path,
                project / "logs" / "debug.jsonl",
            )
            with open(project / "logs" / "debug.jsonl", encoding="utf-8") as handle:
                events = [json.loads(line) for line in handle]

        self.assertEqual(events[0]["type"], "cli_invocation")
        self.assertEqual(events[0]["mode"], "graph")
        self.assertEqual(events[-1]["type"], "cli_finished")

    def test_graph_subcommand_rejects_zero_max_nodes(self):
        with patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["graph", "--max-nodes", "0", "task"])

    def test_run_subcommand_keeps_direct_mode(self):
        with patch("stem_agent.backends.CodexExecBackend.run"):
            with patch("sys.stdout", new=io.StringIO()) as stdout:
                self.assertEqual(main(["run", "hello", "--output-budget", "0"]), 0)

        self.assertIn("stop=output_budget_spent", stdout.getvalue())

    def test_pyproject_declares_stem_agent_console_script(self):
        with open("pyproject.toml", "rb") as handle:
            pyproject = tomllib.load(handle)

        self.assertEqual(
            pyproject["project"]["scripts"]["stem-agent"],
            "stem_agent.cli:main",
        )


@unittest.skipUnless(
    os.environ.get("RUN_CODEX_INTEGRATION") == "1",
    "set RUN_CODEX_INTEGRATION=1 to run real codex exec",
)
class CodexIntegrationTests(unittest.TestCase):
    def test_real_codex_exec_stop(self):
        backend = CodexExecBackend()
        result = backend.run("Say exactly: STOP")

        self.assertTrue(result.session_id)
        self.assertIn("STOP", result.last_text)
        self.assertGreater(result.usage.output_tokens, 0)
        self.assertGreater(result.usage.total_tokens, 0)
        self.assertTrue(result.saw_usage)


if __name__ == "__main__":
    unittest.main()
