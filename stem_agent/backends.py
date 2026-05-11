from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Protocol

from .codex_json import build_codex_command, parse_events
from .models import TurnResult


class Backend(Protocol):
    def run(self, prompt: str, session_id: str | None = None) -> TurnResult:
        ...


@dataclass(frozen=True)
class CodexExecConfig:
    model: str | None = None
    cd: str | None = None
    sandbox: str | None = None


class CodexExecBackend:
    def __init__(
        self,
        *,
        model: str | None = None,
        cd: str | None = None,
        sandbox: str | None = None,
    ) -> None:
        self.config = CodexExecConfig(model=model, cd=cd, sandbox=sandbox)

    @classmethod
    def from_args(cls, args: object) -> "CodexExecBackend":
        return cls(
            model=getattr(args, "model", None),
            cd=getattr(args, "cd", None),
            sandbox=getattr(args, "sandbox", None),
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
        )

    def run(self, prompt: str, session_id: str | None = None) -> TurnResult:
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
