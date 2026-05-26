"""Reusable scenario harness for Kortex demos.

This module contains generic logging, planning, execution, and working-memory
helpers. Scenario modules provide domain manifests, plugins, objects, state,
and scripted turns; the harness owns the repeated runtime mechanics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from unified_planning.plans import Plan

from kortex.config.bootstrapper import DomainBootstrapper
from kortex.memory.adapters import (
    planner_fact_record_from_dict,
    planner_fact_records_from_action_effects,
)
from kortex.memory.working import WorkingMemoryState
from kortex.plugins.registry import PluginRegistry
from kortex.spine.driver import ExecutionDriver
from kortex.spine.planner import KortexPlanner


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
    """Collect structured scenario events and print readable progress."""

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
        self.event(stage=stage, message=message, payload=payload, prefix="trace")

    def event(
        self,
        stage: str,
        message: str,
        payload: dict[str, Any] | None = None,
        prefix: str = "demo",
    ) -> None:
        """Record a structured scenario event and print it."""
        event_payload = _jsonable(payload or {})
        self._require_current().events.append(
            {
                "stage": stage,
                "message": message,
                "payload": event_payload,
            }
        )
        print(f"[{prefix}] {stage}: {message} {json.dumps(event_payload, sort_keys=True)}")

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
        jsonable_results = _jsonable(results)
        self._require_current().results = jsonable_results
        self.note(f"Results: {json.dumps(jsonable_results)}")

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
                "events": _jsonable(log.events),
                "plan": log.plan,
                "results": _jsonable(log.results),
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


def setup_planner(
    domain_path: Path,
    registry: PluginRegistry,
    objects: dict[str, str],
    initial_state: list[dict[str, Any]],
    planner_name: str = "demo_spine",
) -> tuple[KortexPlanner, DomainBootstrapper]:
    """Load a domain and problem state into a fresh planner."""
    planner = KortexPlanner(planner_name)
    bootstrapper = DomainBootstrapper(planner, registry=registry)
    bootstrapper.load_domain(str(domain_path))
    bootstrapper.load_problem_state(objects, initial_state)
    return planner, bootstrapper


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


def execute_plan_with_logging(
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


def _jsonable(value: Any) -> Any:
    """Convert Pydantic and nested runtime values into JSON-safe data."""
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value
