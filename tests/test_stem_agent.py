import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from stem_agent import RunState, TurnResult, Usage
from stem_agent.backends import CodexExecBackend
from stem_agent.cli import main
from stem_agent.codex_json import build_codex_command, parse_events, parse_usage
from stem_agent.graph import END_NODE, parse_node_output, validate_graph, write_graph
from stem_agent.graph_runner import GraphRunner
from stem_agent.policies import BudgetPolicy, StopPolicy


def graph_turn(text, output_tokens=1, saw_usage=True):
    return TurnResult(
        "session",
        text,
        Usage(output_tokens=output_tokens, total_tokens=output_tokens),
        saw_usage,
    )


class FakeBackend:
    def __init__(self, actions):
        self.actions = list(actions)
        self.prompts = []

    def run(self, prompt, session_id=None):
        self.prompts.append(prompt)
        if not self.actions:
            raise RuntimeError("no fake backend actions left")
        action = self.actions.pop(0)
        if callable(action):
            return action(prompt)
        return action


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
                model="gpt-5.2",
                cd="/tmp/x",
                sandbox="read-only",
            ),
            [
                "codex",
                "exec",
                "--json",
                "--model",
                "gpt-5.2",
                "--cd",
                "/tmp/x",
                "--sandbox",
                "read-only",
                "hello",
            ],
        )

    def test_build_resume_codex_command_keeps_options_before_session(self):
        self.assertEqual(
            build_codex_command(
                "hello",
                "session-1",
                model="gpt-5.2",
                cd="/tmp/x",
                sandbox="read-only",
            ),
            [
                "codex",
                "exec",
                "resume",
                "--json",
                "--model",
                "gpt-5.2",
                "session-1",
                "hello",
            ],
        )

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
                "model": "gpt-5.2",
                "effort": "high",
                "prompt": "Fix the graph.",
            },
            "nodes": {
                "work": {
                    "model": "gpt-5.2",
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
                [graph_turn('{"route":"done","result":{"summary":"ok"}}')]
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
                ]
            )

            outcome = GraphRunner(
                graph_path,
                backend_factory=lambda settings: backend,
            ).run("task")

            self.assertEqual(outcome.stop_reason, "graph_finished")
            self.assertEqual(outcome.context[-1]["node"], "finish")
            self.assertTrue(any("node_failed" in prompt for prompt in backend.prompts))

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

    def test_graph_cli_does_not_require_output_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            graph_path = os.path.join(tmpdir, "graph.json")
            write_graph(graph_path, self.valid_graph())
            fake_backend = FakeBackend(
                [graph_turn('{"route":"done","result":{"summary":"ok"}}')]
            )

            with patch("stem_agent.graph_runner.CodexExecBackend", return_value=fake_backend):
                with patch("sys.stdout", new=io.StringIO()) as stdout:
                    self.assertEqual(main(["--graph", graph_path, "task"]), 0)

            self.assertIn('result={"summary":"ok"}', stdout.getvalue())
            self.assertIn("stop=graph_finished", stdout.getvalue())


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
