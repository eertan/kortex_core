"""
Metacognitive novelty branch for total planning impasses.

The novelty branch delegates missing deterministic knowledge to the Pi coding
agent. Its first target is new YAML HTN knowledge using existing primitives;
new Python plugins are reserved for genuinely missing physical capabilities.
"""

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class NoveltyCommandResult:
    """Result returned by a novelty command runner."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


class NoveltyCommandRunner(Protocol):
    """Boundary for invoking external novelty tools."""

    def run(self, command: list[str], cwd: str) -> NoveltyCommandResult:
        """Execute a novelty command in the given workspace."""
        ...


class SubprocessNoveltyRunner:
    """Subprocess-backed runner for the Pi CLI."""

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


class NoveltyBranch:
    """
    Handles total domain impasses (Tier 3).

    When HTN expansion and classical fallback planning both fail, this branch
    constructs a constrained Pi agent task to synthesize missing deterministic
    domain knowledge.
    """

    def __init__(
        self,
        workspace_path: str = "./",
        command_runner: NoveltyCommandRunner | None = None,
        execute_pi: bool = False,
        validation_command: list[str] | None = None,
    ) -> None:
        """Initialize the novelty branch."""
        self.workspace_path = workspace_path
        self.plugins_dir = os.path.join(self.workspace_path, "kortex", "plugins")
        self.domain_manifest_path = os.path.join(self.workspace_path, "domain.yaml")
        self.command_runner = command_runner or SubprocessNoveltyRunner()
        self.execute_pi = execute_pi
        self.validation_command = validation_command
        self.last_prompt: str | None = None
        self.last_command: list[str] | None = None

    def resolve_impasse(
        self,
        failed_goal: dict[str, Any],
        current_state: dict[Any, Any],
        available_actions: list[str],
    ) -> bool:
        """Delegate an impasse to Pi and optionally run validation afterward."""
        print(f"[NoveltyBranch] IMPASSE DETECTED for goal: {failed_goal['fluent']}")

        prompt = self._build_prompt(failed_goal, current_state, available_actions)
        command = ["pi", "run", "worker", prompt]
        self.last_prompt = prompt
        self.last_command = command

        print("[NoveltyBranch] Prepared Pi Agent command.")
        if not self.execute_pi:
            print("[NoveltyBranch] Dry run mode enabled; Pi command not executed.")
            return True

        result = self.command_runner.run(command, cwd=self.workspace_path)
        if result.returncode != 0:
            print(f"[NoveltyBranch] Pi Agent failed: {result.stderr}")
            return False

        if self.validation_command is None:
            return True

        validation_result = self.command_runner.run(
            self.validation_command,
            cwd=self.workspace_path,
        )
        if validation_result.returncode != 0:
            print(f"[NoveltyBranch] Validation failed: {validation_result.stderr}")
            return False

        return True

    def _build_prompt(
        self,
        failed_goal: dict[str, Any],
        current_state: dict[Any, Any],
        available_actions: list[str],
    ) -> str:
        """Build the constrained Pi worker prompt for an impasse."""
        true_fluents = [str(key) for key, value in current_state.items() if value]
        return (
            "The Kortex Core planner has hit a Tier 3 impasse.\n"
            f"Failed goal: {json.dumps(failed_goal)}\n"
            f"Current true state fluents: {json.dumps(true_fluents)}\n"
            f"Available primitive actions in registry: {json.dumps(available_actions)}\n\n"
            "Instructions:\n"
            "1. First determine whether the available primitives can be arranged "
            "into a new deterministic HTN method.\n"
            f"2. If existing primitives are sufficient, edit `{self.domain_manifest_path}` "
            "and add the method under `htn_methods:`.\n"
            "3. Only if a primitive physical capability is missing, add a new safe "
            f"Python plugin under `{self.plugins_dir}` using `@registry.register_action`, "
            "then update the domain manifest action declaration.\n"
            "4. Do not add LLM reasoning into operators, methods, or execution code.\n"
            "5. Keep changes minimal and deterministic, then stop."
        )
