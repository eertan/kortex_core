"""Executable scenario demo for inspecting Kortex planning and execution logs."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from unified_planning.shortcuts import get_environment
from unified_planning.plans import Plan

from kortex.config.bootstrapper import DomainBootstrapper
from kortex.memory.adapters import (
    planner_fact_record_from_dict,
    planner_fact_records_from_action_effects,
)
from kortex.memory.reflector import SleepReflector, SynthesizedMetaTask
from kortex.memory.working import WorkingMemoryState
from kortex.plugins.registry import PluginRegistry
from kortex.sandbox.chunker import IntraDomainLearner
from kortex.sandbox.novelty import NoveltyBranch
from kortex.spine.driver import ExecutionDriver
from kortex.spine.planner import KortexPlanner


BASE_YAML = """
domain_name: "scenario_domain"
types:
  - Location
  - Server
fluents:
  robot_at:
    signature: { loc: Location }
  door_unlocked:
    signature: { loc: Location }
  server_unreachable:
    signature: { srv: Server }
  storage_checked:
    signature: { srv: Server }
  cache_cleared:
    signature: { srv: Server }
  audit_completed:
    signature: { srv: Server }
actions:
  - name: move
    parameters: { frm: Location, to: Location }
    preconditions:
      - fluent: robot_at
        args: [frm]
    effects:
      - fluent: robot_at
        args: [frm]
        value: false
      - fluent: robot_at
        args: [to]
  - name: unlock
    parameters: { loc: Location }
    preconditions:
      - fluent: robot_at
        args: [loc]
    effects:
      - fluent: door_unlocked
        args: [loc]
  - name: wipe_server
    parameters: { srv: Server }
    preconditions: []
    effects:
      - fluent: server_unreachable
        args: [srv]
        value: true
  - name: check_disk_space
    parameters: { srv: Server }
    preconditions: []
    effects:
      - fluent: storage_checked
        args: [srv]
        value: true
  - name: clear_cache
    parameters: { srv: Server }
    preconditions: []
    effects:
      - fluent: cache_cleared
        args: [srv]
        value: true
htn_methods:
  - name: m_deliver_straight
    target_task: "deliver_straight"
    ordered_subtasks:
      - ["move", "frm", "to"]
  - name: m_wipe
    target_task: "wipe_server_task"
    ordered_subtasks:
      - ["wipe_server", "srv"]
"""


@dataclass
class ScenarioLog:
    """Structured record for one demo scenario."""

    scenario: str
    summary: str
    domain_path: str
    events: list[dict[str, Any]] = field(default_factory=list)
    plan: list[dict[str, Any]] = field(default_factory=list)
    results: list[Any] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class DemoLogger:
    """Collects structured events while also printing readable scenario output."""

    def __init__(self) -> None:
        """Initialize an empty scenario log buffer."""
        self.scenario_logs: list[ScenarioLog] = []
        self.current: ScenarioLog | None = None

    def start(self, name: str, summary: str, domain_path: Path) -> None:
        """Start recording a scenario."""
        self.current = ScenarioLog(
            scenario=name,
            summary=summary,
            domain_path=str(domain_path),
        )
        self.scenario_logs.append(self.current)
        print(f"\n=== {name}: {summary} ===")
        self.note(f"Domain manifest: {domain_path}")

    def trace(self, stage: str, message: str, payload: dict[str, Any]) -> None:
        """Record a trace callback event emitted by the execution driver."""
        self._require_current().events.append(
            {
                "stage": stage,
                "message": message,
                "payload": payload,
            }
        )
        print(f"[trace] {stage}: {message} {json.dumps(payload, sort_keys=True)}")

    def record_plan(self, plan: Plan | None) -> None:
        """Record and print the symbolic plan."""
        current = self._require_current()
        if plan is None:
            current.plan = []
            self.note("Planner returned no plan.")
            return

        current.plan = [
            {
                "action": action_instance.action.name,
                "parameters": {
                    parameter.name: actual_value.object().name
                    for parameter, actual_value in zip(
                        action_instance.action.parameters,
                        action_instance.actual_parameters,
                        strict=True,
                    )
                },
            }
            for action_instance in plan.actions
        ]
        self.note(f"Plan: {json.dumps(current.plan, sort_keys=True)}")

    def record_results(self, results: list[Any]) -> None:
        """Record and print physical execution results."""
        self._require_current().results = results
        self.note(f"Results: {json.dumps(results)}")

    def note(self, message: str) -> None:
        """Append and print a human-readable scenario note."""
        self._require_current().notes.append(message)
        print(f"[demo] {message}")

    def write_json(self, output_path: Path) -> None:
        """Persist all scenario logs as JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "scenario": log.scenario,
                "summary": log.summary,
                "domain_path": log.domain_path,
                "events": log.events,
                "plan": log.plan,
                "results": log.results,
                "notes": log.notes,
            }
            for log in self.scenario_logs
        ]
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n[demo] Wrote structured log: {output_path}")

    def _require_current(self) -> ScenarioLog:
        """Return the active scenario log or fail if no scenario is running."""
        if self.current is None:
            raise RuntimeError("No active scenario log.")
        return self.current


def build_registry() -> PluginRegistry:
    """Create an isolated plugin registry for demo scenarios."""
    registry = PluginRegistry()

    @registry.register_action("move")
    def move(frm: str, to: str) -> str:
        return f"Moved from {frm} to {to}"

    @registry.register_action("unlock")
    def unlock(loc: str) -> str:
        return f"Unlocked {loc}"

    @registry.register_action("wipe_server", requires_approval=True)
    def wipe_server(srv: str) -> str:
        return f"WIPED DATA ON {srv}"

    @registry.register_action("check_disk_space")
    def check_disk_space(srv: str) -> str:
        return f"Checked disk space on {srv}"

    @registry.register_action("clear_cache")
    def clear_cache(srv: str) -> str:
        return f"Cleared cache on {srv}"

    return registry


def setup_domain(workspace: Path) -> Path:
    """Create a temporary scenario domain manifest."""
    domain_path = workspace / "domain.yaml"
    domain_path.write_text(BASE_YAML, encoding="utf-8")
    return domain_path


def setup_planner(
    domain_path: Path,
    registry: PluginRegistry,
    objects: dict[str, str],
    initial_state: list[dict[str, Any]],
) -> tuple[KortexPlanner, DomainBootstrapper]:
    """Load the domain and state into a fresh planner."""
    planner = KortexPlanner("demo_spine")
    bootstrapper = DomainBootstrapper(planner, registry=registry)
    bootstrapper.load_domain(str(domain_path))
    bootstrapper.load_problem_state(objects, initial_state)
    return planner, bootstrapper


def has_learned_method(domain_path: Path, target_task: str) -> bool:
    """Return whether the domain manifest contains a specific learned HTN method."""
    import yaml

    manifest = yaml.safe_load(domain_path.read_text(encoding="utf-8")) or {}
    return any(
        method.get("target_task") == target_task
        for method in manifest.get("htn_methods", [])
    )


def get_learned_method(domain_path: Path, target_task: str) -> dict[str, Any] | None:
    """Return a learned method definition from the domain manifest."""
    import yaml

    manifest = yaml.safe_load(domain_path.read_text(encoding="utf-8")) or {}
    for method in manifest.get("htn_methods", []):
        if method.get("target_task") == target_task:
            return method
    return None


def create_access_secure_vault_goal(
    bootstrapper: DomainBootstrapper,
    domain_path: Path,
    logger: DemoLogger,
) -> None:
    """Create the same access request, using learned HTN when available."""
    request = {
        "intent": "access_secure_vault",
        "frm": "lobby",
        "to": "vault",
    }
    logger.note(f"Request: {json.dumps(request, sort_keys=True)}")

    if has_learned_method(domain_path, "access_secure_vault"):
        logger.note(
            "Dispatch: learned HTN method found; using direct deterministic expansion "
            "instead of classical planner search."
        )
        bootstrapper.create_goal(
            {"task": "access_secure_vault", "args": [request["frm"], request["to"]]}
        )
        return

    logger.note(
        "Dispatch: no learned method found; using classical planner search for the same request."
    )
    bootstrapper.create_goal({"fluent": "robot_at", "args": [request["to"]]})
    bootstrapper.create_goal({"fluent": "door_unlocked", "args": [request["to"]]})


def execute(
    plan: Plan,
    registry: PluginRegistry,
    logger: DemoLogger,
    bootstrapper: DomainBootstrapper | None = None,
    working_memory: WorkingMemoryState | None = None,
    interactive: bool = False,
) -> list[Any]:
    """Execute a plan through the physical driver with trace capture."""
    driver = ExecutionDriver(
        interactive=interactive,
        registry=registry,
        trace_callback=logger.trace,
    )
    results = driver.execute_plan(plan)
    if bootstrapper is not None and working_memory is not None:
        apply_plan_effects_to_working_memory(plan, bootstrapper, working_memory)
        logger.note(
            "Working memory facts: "
            + json.dumps(working_memory.to_bootstrapper_initial_state(), sort_keys=True)
        )
    logger.record_results(results)
    return results


def build_working_memory(
    session_id: str,
    initial_state: list[dict[str, Any]],
) -> WorkingMemoryState:
    """Create and seed demo working memory from explicit initial facts."""
    working_memory = WorkingMemoryState(session_id=session_id)
    for fact in initial_state:
        record = planner_fact_record_from_dict(
            fact,
            source_system="scenario_initial_state",
            source_reference=session_id,
        )
        working_memory.hydrate_planner_fact(record)
    return working_memory


def apply_plan_effects_to_working_memory(
    plan: Plan,
    bootstrapper: DomainBootstrapper,
    working_memory: WorkingMemoryState,
) -> None:
    """Apply declared action effects into demo working memory."""
    for action_instance in plan.actions:
        for record in planner_fact_records_from_action_effects(
            action_instance,
            bootstrapper.action_specs,
            source_system="scenario_execution_effects",
            source_reference=working_memory.session_id,
        ):
            working_memory.hydrate_planner_fact(record)


def scenario_1(domain_path: Path, registry: PluginRegistry, logger: DemoLogger) -> None:
    """Run direct HTN expansion for a perfectly specified task."""
    logger.start("scenario_1", "direct HTN method execution", domain_path)
    initial_state = [{"fluent": "robot_at", "args": ["lobby"]}]
    working_memory = build_working_memory("scenario_1", initial_state)
    planner, bootstrapper = setup_planner(
        domain_path,
        registry,
        {"lobby": "Location", "hallway": "Location"},
        initial_state,
    )
    bootstrapper.create_goal({"task": "deliver_straight", "args": ["lobby", "hallway"]})
    plan = planner.execute_plan()
    logger.record_plan(plan)
    if plan is not None:
        execute(plan, registry, logger, bootstrapper, working_memory)


def scenario_2(domain_path: Path, registry: PluginRegistry, logger: DemoLogger) -> None:
    """Run classical planning once, then save the successful trace as a skill."""
    logger.start("scenario_2", "classical planner bridges a goal gap and saves a skill", domain_path)
    initial_state = [
        {"fluent": "robot_at", "args": ["lobby"]},
        {"fluent": "door_unlocked", "args": ["lobby"]},
    ]
    working_memory = build_working_memory("scenario_2", initial_state)
    planner, bootstrapper = setup_planner(
        domain_path,
        registry,
        {"lobby": "Location", "vault": "Location"},
        initial_state,
    )
    create_access_secure_vault_goal(bootstrapper, domain_path, logger)
    plan = planner.execute_plan()
    logger.record_plan(plan)
    if plan is not None:
        execute(plan, registry, logger, bootstrapper, working_memory)
        chunker = IntraDomainLearner(str(domain_path))
        chunker.chunk_successful_plan(
            failed_task_name="access_secure_vault",
            preconditions={},
            plan_actions=[["move", "frm", "to"], ["unlock", "to"]],
        )
        logger.note(
            "Saved skill: access_secure_vault -> move(frm, to), unlock(to). "
            f"Manifest updated at {domain_path}"
        )
        learned_method = get_learned_method(domain_path, "access_secure_vault")
        if learned_method is not None:
            logger.note(
                "Saved skill contract: "
                + json.dumps(
                    {
                        "parameters": learned_method.get("parameters", {}),
                        "preconditions": learned_method.get("preconditions", []),
                        "effects": learned_method.get("effects", []),
                    },
                    sort_keys=True,
                )
            )


def scenario_3(domain_path: Path, registry: PluginRegistry, logger: DemoLogger) -> None:
    """Run the same access request through the skill saved by scenario 2."""
    logger.start("scenario_3", "same request executes through the saved HTN skill", domain_path)
    if not has_learned_method(domain_path, "access_secure_vault"):
        logger.note(
            "No saved skill was present. Seeding access_secure_vault so this scenario "
            "can still run by itself."
        )
        chunker = IntraDomainLearner(str(domain_path))
        chunker.chunk_successful_plan(
            failed_task_name="access_secure_vault",
            preconditions={},
            plan_actions=[["move", "frm", "to"], ["unlock", "to"]],
        )
    initial_state = [{"fluent": "robot_at", "args": ["lobby"]}]
    working_memory = build_working_memory("scenario_3", initial_state)
    planner, bootstrapper = setup_planner(
        domain_path,
        registry,
        {"lobby": "Location", "vault": "Location"},
        initial_state,
    )
    create_access_secure_vault_goal(bootstrapper, domain_path, logger)
    plan = planner.execute_plan()
    logger.record_plan(plan)
    if plan is not None:
        execute(plan, registry, logger, bootstrapper, working_memory)


def scenario_4(domain_path: Path, registry: PluginRegistry, logger: DemoLogger) -> None:
    """Run HITL denial and approval paths for a sensitive primitive."""
    logger.start("scenario_4", "HITL approval gates a sensitive action", domain_path)
    initial_state = [{"fluent": "server_unreachable", "args": ["prod_db"], "value": False}]
    working_memory = build_working_memory("scenario_4", initial_state)
    planner, bootstrapper = setup_planner(
        domain_path,
        registry,
        {"prod_db": "Server"},
        initial_state,
    )
    bootstrapper.create_goal({"fluent": "server_unreachable", "args": ["prod_db"]})
    plan = planner.execute_plan()
    logger.record_plan(plan)
    if plan is None:
        return

    driver = ExecutionDriver(
        interactive=True,
        registry=registry,
        trace_callback=logger.trace,
    )
    with patch("builtins.input", return_value="n"):
        try:
            driver.execute_plan(plan)
        except PermissionError as error:
            logger.note(f"Denied run stopped as expected: {error}")

    with patch("builtins.input", return_value="y"):
        results = driver.execute_plan(plan)
    apply_plan_effects_to_working_memory(plan, bootstrapper, working_memory)
    logger.note(
        "Working memory facts: "
        + json.dumps(working_memory.to_bootstrapper_initial_state(), sort_keys=True)
    )
    logger.record_results(results)


def scenario_5(domain_path: Path, registry: PluginRegistry, logger: DemoLogger) -> None:
    """Run novelty branch routing for a total planning impasse."""
    logger.start("scenario_5", "planner impasse routes to novelty branch", domain_path)
    planner, bootstrapper = setup_planner(
        domain_path,
        registry,
        {"prod_db": "Server"},
        [],
    )
    failed_goal = {"fluent": "audit_completed", "args": ["prod_db"], "value": True}
    bootstrapper.create_goal(failed_goal)
    plan = planner.execute_plan()
    logger.record_plan(plan)

    novelty = NoveltyBranch(workspace_path=str(domain_path.parent))
    result = novelty.resolve_impasse_result(
        failed_goal=failed_goal,
        current_state={},
        available_actions=registry.list_plugins(),
    )
    logger.note(
        "Novelty result: "
        + json.dumps(
            {
                "resolved": result.resolved,
                "provider": result.provider,
                "message": result.message,
                "command": result.command,
            },
            sort_keys=True,
        )
    )


def scenario_6(domain_path: Path, registry: PluginRegistry, logger: DemoLogger) -> None:
    """Run sleep reflection synthesis and execute the produced meta-task."""
    logger.start("scenario_6", "sleep reflection creates an executable meta-task", domain_path)
    os.environ.pop("GOOGLE_API_KEY", None)
    os.environ["GEMINI_API_KEY"] = "fake-key"
    reflector = SleepReflector(str(domain_path))
    reflector.client.models.generate_content = MagicMock(
        return_value=SynthesizedMetaTask(
            meta_task_name="optimize_local_storage",
            preconditions={"server_reachable": True},
            ordered_subtasks=[
                ["check_disk_space", "srv"],
                ["clear_cache", "srv"],
            ],
        )
    )

    reflected = reflector.reflect_and_synthesize(
        [
            ["ssh_login", "check_disk_space", "clear_cache", "reindex_table"],
            ["check_disk_space", "clear_cache", "restart_service"],
            ["download_logs", "check_disk_space", "clear_cache"],
        ]
    )
    logger.note(f"Reflection synthesized method: {reflected}")

    planner, bootstrapper = setup_planner(
        domain_path,
        registry,
        {"prod_db": "Server"},
        [],
    )
    working_memory = build_working_memory("scenario_6", [])
    bootstrapper.create_goal({"task": "optimize_local_storage", "args": ["prod_db"]})
    plan = planner.execute_plan()
    logger.record_plan(plan)
    if plan is not None:
        execute(plan, registry, logger, bootstrapper, working_memory)


SCENARIOS: dict[str, Callable[[Path, PluginRegistry, DemoLogger], None]] = {
    "1": scenario_1,
    "2": scenario_2,
    "3": scenario_3,
    "4": scenario_4,
    "5": scenario_5,
    "6": scenario_6,
}


def parse_args() -> argparse.Namespace:
    """Parse demo CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run Kortex scenario demos with readable and structured logs."
    )
    parser.add_argument(
        "--scenario",
        choices=["all", *SCENARIOS.keys()],
        default="all",
        help="Scenario number to run, or all.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("demo_logs/scenario_demo_latest.json"),
        help="Path for the structured JSON log.",
    )
    return parser.parse_args()


def main() -> None:
    """Run selected demo scenarios."""
    args = parse_args()
    get_environment().credits_stream = None
    logger = DemoLogger()
    registry = build_registry()
    selected = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]

    with tempfile.TemporaryDirectory(prefix="kortex_scenario_demo_") as temp_dir:
        learned_access_domain: Path | None = None
        for scenario_id in selected:
            if scenario_id == "3" and learned_access_domain is not None:
                domain_path = learned_access_domain
            else:
                scenario_workspace = Path(temp_dir) / f"scenario_{scenario_id}"
                scenario_workspace.mkdir(parents=True, exist_ok=True)
                domain_path = setup_domain(scenario_workspace)
            SCENARIOS[scenario_id](domain_path, registry, logger)
            if scenario_id == "2":
                learned_access_domain = domain_path

    logger.write_json(args.log_path)


if __name__ == "__main__":
    main()
