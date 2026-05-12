from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from typing import Protocol

from .codex_json import (
    CodexEventAccumulator,
    build_codex_command,
    parse_event_line,
    parse_events,
)
from .debug_log import DebugLog
from .models import TurnResult
from .progress import SingleLineProgress


class Backend(Protocol):
    def run(self, prompt: str, session_id: str | None = None) -> TurnResult:
        ...


@dataclass(frozen=True)
class CodexExecConfig:
    model: str | None = None
    cd: str | None = None
    sandbox: str | None = None
    skip_git_repo_check: bool = False
    auto_review: bool = False
    config_overrides: tuple[str, ...] = ()


class CodexExecBackend:
    def __init__(
        self,
        *,
        model: str | None = None,
        cd: str | None = None,
        sandbox: str | None = None,
        skip_git_repo_check: bool = False,
        auto_review: bool = False,
        config_overrides: list[str] | tuple[str, ...] = (),
        progress: SingleLineProgress | None = None,
        debug_log: DebugLog | None = None,
        debug_label: str = "codex",
    ) -> None:
        self.config = CodexExecConfig(
            model=model,
            cd=cd,
            sandbox=sandbox,
            skip_git_repo_check=skip_git_repo_check,
            auto_review=auto_review,
            config_overrides=tuple(config_overrides),
        )
        self.progress = progress
        self.debug_log = debug_log
        self.debug_label = debug_label

    @classmethod
    def from_args(
        cls,
        args: object,
        *,
        debug_log: DebugLog | None = None,
        debug_label: str = "codex",
    ) -> "CodexExecBackend":
        return cls(
            model=getattr(args, "model", None),
            cd=getattr(args, "cd", None),
            sandbox=getattr(args, "sandbox", None),
            skip_git_repo_check=getattr(args, "skip_git_repo_check", False),
            auto_review=getattr(args, "auto_review", False),
            debug_log=debug_log,
            debug_label=debug_label,
        )

    def build_codex_command(
        self,
        prompt: str,
        session_id: str | None = None,
    ) -> list[str]:
        return build_codex_command(
            prompt,
            session_id,
            model=self.config.model,
            cd=self.config.cd,
            sandbox=self.config.sandbox,
            skip_git_repo_check=self.config.skip_git_repo_check,
            auto_review=self.config.auto_review,
            config_overrides=self.config.config_overrides,
        )

    def run(self, prompt: str, session_id: str | None = None) -> TurnResult:
        if self.progress is not None:
            return self._run_streaming(prompt, session_id)

        command = self.build_codex_command(prompt, session_id)
        self._write_debug(
            "codex_exec_started",
            command=command,
            config=asdict(self.config),
            prompt=prompt,
            session_id=session_id,
        )
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            input="",
            cwd=self.config.cd,
        )
        self._write_debug(
            "codex_exec_completed",
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip()
            self._write_debug("codex_exec_failed", message=message)
            raise RuntimeError(message)
        try:
            parsed = parse_events(result.stdout)
        except Exception as error:
            self._write_debug("codex_parse_failed", error=str(error))
            raise
        self._write_debug_result(parsed)
        return parsed

    def _run_streaming(
        self,
        prompt: str,
        session_id: str | None = None,
    ) -> TurnResult:
        command = self.build_codex_command(prompt, session_id)
        self._write_debug(
            "codex_exec_started",
            command=command,
            config=asdict(self.config),
            prompt=prompt,
            session_id=session_id,
        )
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            cwd=self.config.cd,
        )
        self._write_debug("codex_process_started", pid=getattr(process, "pid", None))
        accumulator = CodexEventAccumulator()
        stdout_lines: list[str] = []
        assert self.progress is not None
        self.progress.start()

        try:
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                stdout_lines.append(line)
                self._write_debug("codex_stdout_line", line=line)
                try:
                    event = parse_event_line(line)
                except Exception as error:
                    self._write_debug("codex_parse_failed", line=line, error=str(error))
                    raise
                self._write_debug("codex_event", event=event)
                accumulator.add(event)
                self.progress.event(str(event.get("type", "-")))

            returncode = process.wait()
            stderr = process.stderr.read() if process.stderr is not None else ""
            self._write_debug(
                "codex_exec_completed",
                returncode=returncode,
                stdout_line_count=len(stdout_lines),
                stderr=stderr,
            )
            if returncode != 0:
                message = stderr.strip() or "\n".join(stdout_lines).strip()
                self._write_debug("codex_exec_failed", message=message)
                self.progress.failed(message)
                raise RuntimeError(message)

            result = accumulator.result()
            self._write_debug_result(result)
            self.progress.finish(result)
            return result
        except Exception as error:
            if process.poll() is None:
                process.kill()
                process.wait()
                self._write_debug("codex_process_killed")
            self._write_debug("codex_exec_exception", error=str(error))
            if not isinstance(error, RuntimeError):
                self.progress.failed(str(error))
            raise

    def _write_debug(self, event_type: str, **fields: object) -> None:
        if self.debug_log is None:
            return
        self.debug_log.write(event_type, label=self.debug_label, **fields)

    def _write_debug_result(self, result: TurnResult) -> None:
        self._write_debug(
            "codex_result_parsed",
            session_id=result.session_id,
            saw_usage=result.saw_usage,
            usage=result.usage.__dict__,
            last_text=result.last_text,
            raw_event_count=len(result.raw_events),
        )
