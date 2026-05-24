"""
Provider-neutral novelty branch for total planning impasses.

Kortex core depends on the NoveltyAgent protocol, not on a specific coding
agent provider. Pi, Codex, LangChain DeepAgents, or local CLIs can implement
the same request/result contract.
"""

import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class NoveltyRequest:
    """Provider-neutral impasse payload for novelty treatment."""

    failed_goal: dict[str, Any]
    current_state: dict[Any, Any]
    available_actions: list[str]
    workspace_path: str
    domain_manifest_path: str
    plugins_dir: str
    validation_command: list[str] | None = None
    constraints: list[str] = field(
        default_factory=lambda: [
            "Prefer adding deterministic HTN methods using existing primitives.",
            "Only create Python plugins when a physical primitive is missing.",
            "Do not add LLM reasoning into normal execution operators.",
            "Keep changes minimal and validate before reporting success.",
        ]
    )


@dataclass(frozen=True)
class NoveltyResult:
    """Provider-neutral novelty resolution result."""

    resolved: bool
    provider: str
    message: str
    changed_files: list[str] = field(default_factory=list)
    command: list[str] | None = None
    prompt: str | None = None
    validation_passed: bool | None = None
    stdout: str = ""
    stderr: str = ""


class NoveltyAgent(Protocol):
    """Backend protocol for novelty providers."""

    provider_name: str

    def resolve(self, request: NoveltyRequest) -> NoveltyResult:
        """Attempt to resolve a deterministic planning impasse."""
        ...


@dataclass(frozen=True)
class NoveltyCommandResult:
    """Result returned by a command runner."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


class NoveltyCommandRunner(Protocol):
    """Boundary for invoking external novelty tools."""

    def run(self, command: list[str], cwd: str) -> NoveltyCommandResult:
        """Execute a command in the given workspace."""
        ...


class SubprocessNoveltyRunner:
    """Subprocess-backed command runner for CLI novelty providers."""

    def run(self, command: list[str], cwd: str) -> NoveltyCommandResult:
        """Run the command and capture output."""
        process = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
        return NoveltyCommandResult(
            returncode=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
        )


class PiNoveltyAgent:
    """Novelty provider backed by the Pi agent CLI."""

    provider_name = "pi"

    def __init__(
        self,
        command_runner: NoveltyCommandRunner | None = None,
        execute: bool = False,
    ) -> None:
        """Initialize the Pi provider."""
        self.command_runner = command_runner or SubprocessNoveltyRunner()
        self.execute = execute

    def resolve(self, request: NoveltyRequest) -> NoveltyResult:
        """Resolve an impasse by delegating to `pi run worker`."""
        prompt = build_novelty_prompt(request)
        command = ["pi", "run", "worker", prompt]

        if not self.execute:
            return NoveltyResult(
                resolved=True,
                provider=self.provider_name,
                message="Dry run mode enabled; Pi command was prepared but not executed.",
                command=command,
                prompt=prompt,
            )

        result = self.command_runner.run(command, cwd=request.workspace_path)
        if result.returncode != 0:
            return NoveltyResult(
                resolved=False,
                provider=self.provider_name,
                message="Pi provider failed.",
                command=command,
                prompt=prompt,
                stdout=result.stdout,
                stderr=result.stderr,
            )

        validation_passed: bool | None = None
        if request.validation_command is not None:
            validation = self.command_runner.run(
                request.validation_command,
                cwd=request.workspace_path,
            )
            validation_passed = validation.returncode == 0
            if not validation_passed:
                return NoveltyResult(
                    resolved=False,
                    provider=self.provider_name,
                    message="Pi provider completed but validation failed.",
                    command=command,
                    prompt=prompt,
                    validation_passed=False,
                    stdout=validation.stdout,
                    stderr=validation.stderr,
                )

        return NoveltyResult(
            resolved=True,
            provider=self.provider_name,
            message="Pi provider resolved impasse.",
            command=command,
            prompt=prompt,
            validation_passed=validation_passed,
            stdout=result.stdout,
            stderr=result.stderr,
        )


class NoveltyBranch:
    """
    Provider-neutral novelty branch facade.

    The branch creates a NoveltyRequest and delegates it to the configured
    NoveltyAgent backend. `PiNoveltyAgent` is the default backend for now.
    """

    def __init__(
        self,
        workspace_path: str = "./",
        novelty_agent: NoveltyAgent | None = None,
        command_runner: NoveltyCommandRunner | None = None,
        execute_pi: bool = False,
        validation_command: list[str] | None = None,
    ) -> None:
        """Initialize the novelty branch facade."""
        self.workspace_path = workspace_path
        self.plugins_dir = os.path.join(self.workspace_path, "kortex", "plugins")
        self.domain_manifest_path = os.path.join(self.workspace_path, "domain.yaml")
        self.validation_command = validation_command
        self.novelty_agent = novelty_agent or PiNoveltyAgent(
            command_runner=command_runner,
            execute=execute_pi,
        )
        self.last_request: NoveltyRequest | None = None
        self.last_result: NoveltyResult | None = None

    def resolve_impasse(
        self,
        failed_goal: dict[str, Any],
        current_state: dict[Any, Any],
        available_actions: list[str],
    ) -> bool:
        """Resolve an impasse through the configured novelty provider."""
        result = self.resolve_impasse_result(
            failed_goal=failed_goal,
            current_state=current_state,
            available_actions=available_actions,
        )
        return result.resolved

    def resolve_impasse_result(
        self,
        failed_goal: dict[str, Any],
        current_state: dict[Any, Any],
        available_actions: list[str],
    ) -> NoveltyResult:
        """Resolve an impasse and return provider-neutral result metadata."""
        print(f"[NoveltyBranch] IMPASSE DETECTED for goal: {failed_goal['fluent']}")
        request = NoveltyRequest(
            failed_goal=failed_goal,
            current_state=current_state,
            available_actions=available_actions,
            workspace_path=self.workspace_path,
            domain_manifest_path=self.domain_manifest_path,
            plugins_dir=self.plugins_dir,
            validation_command=self.validation_command,
        )
        self.last_request = request
        result = self.novelty_agent.resolve(request)
        self.last_result = result
        return result


def build_novelty_prompt(request: NoveltyRequest) -> str:
    """Build a constrained provider prompt for novelty resolution."""
    true_fluents = [str(key) for key, value in request.current_state.items() if value]
    constraints = "\n".join(
        f"{index}. {constraint}"
        for index, constraint in enumerate(request.constraints, start=1)
    )
    return (
        "The Kortex Core planner has hit a Tier 3 impasse.\n"
        f"Failed goal: {json.dumps(request.failed_goal)}\n"
        f"Current true state fluents: {json.dumps(true_fluents)}\n"
        f"Available primitive actions in registry: {json.dumps(request.available_actions)}\n"
        f"Domain manifest path: `{request.domain_manifest_path}`\n"
        f"Plugin directory: `{request.plugins_dir}`\n\n"
        "Provider constraints:\n"
        f"{constraints}\n\n"
        "Instructions:\n"
        "1. First determine whether the available primitives can be arranged "
        "into a new deterministic HTN method.\n"
        "2. If existing primitives are sufficient, edit the domain manifest and "
        "add the method under `htn_methods:`.\n"
        "3. Only if a primitive physical capability is missing, add a new safe "
        "Python plugin using `@registry.register_action`, then update the "
        "domain manifest action declaration.\n"
        "4. Do not add LLM reasoning into operators, methods, or execution code.\n"
        "5. Keep changes minimal and deterministic, then stop."
    )
