from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Protocol

from .codex_json import (
    CodexEventAccumulator,
    build_codex_command,
    parse_event_line,
    parse_events,
)
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
    config_overrides: tuple[str, ...] = ()


class CodexExecBackend:
    def __init__(
        self,
        *,
        model: str | None = None,
        cd: str | None = None,
        sandbox: str | None = None,
        skip_git_repo_check: bool = False,
        config_overrides: list[str] | tuple[str, ...] = (),
        progress: SingleLineProgress | None = None,
    ) -> None:
        self.config = CodexExecConfig(
            model=model,
            cd=cd,
            sandbox=sandbox,
            skip_git_repo_check=skip_git_repo_check,
            config_overrides=tuple(config_overrides),
        )
        self.progress = progress

    @classmethod
    def from_args(cls, args: object) -> "CodexExecBackend":
        return cls(
            model=getattr(args, "model", None),
            cd=getattr(args, "cd", None),
            sandbox=getattr(args, "sandbox", None),
            skip_git_repo_check=getattr(args, "skip_git_repo_check", False),
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
            config_overrides=self.config.config_overrides,
        )

    def run(self, prompt: str, session_id: str | None = None) -> TurnResult:
        if self.progress is not None:
            return self._run_streaming(prompt, session_id)

        result = subprocess.run(
            self.build_codex_command(prompt, session_id),
            text=True,
            capture_output=True,
            check=False,
            input="",
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return parse_events(result.stdout)

    def _run_streaming(
        self,
        prompt: str,
        session_id: str | None = None,
    ) -> TurnResult:
        process = subprocess.Popen(
            self.build_codex_command(prompt, session_id),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
        )
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
                event = parse_event_line(line)
                accumulator.add(event)
                self.progress.event(str(event.get("type", "-")))

            returncode = process.wait()
            stderr = process.stderr.read() if process.stderr is not None else ""
            if returncode != 0:
                message = stderr.strip() or "\n".join(stdout_lines).strip()
                self.progress.failed(message)
                raise RuntimeError(message)

            result = accumulator.result()
            self.progress.finish(result)
            return result
        except Exception as error:
            if process.poll() is None:
                process.kill()
                process.wait()
            if not isinstance(error, RuntimeError):
                self.progress.failed(str(error))
            raise
