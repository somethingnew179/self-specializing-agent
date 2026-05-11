import io
import os
import tempfile
import unittest
from unittest.mock import patch

from stem_agent import RunState, TurnResult, Usage
from stem_agent.backends import CodexExecBackend
from stem_agent.cli import main
from stem_agent.codex_json import build_codex_command, parse_events, parse_usage
from stem_agent.policies import BudgetPolicy, StopPolicy


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
                self.assertEqual(main(["hello", "--output-budget", "100", "--max-turns", "1"]), 0)

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
