"""
Top-level Kortex Core agent runtime.

This module composes the statistical extraction boundary, optional memory
hydration, deterministic planning, physical execution, tracing, and episodic
writeback into one explicit run loop.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import uuid4

from unified_planning.plans import Plan

from kortex.config.bootstrapper import DomainBootstrapper
from kortex.extractor.models import ClarificationRequired, HTNLaunchPad, IntentExtraction
from kortex.memory.manager import MemoryManager
from kortex.spine.driver import ExecutionDriver
from kortex.spine.planner import KortexPlanner
from kortex.tracing import TraceEvent, TraceRecorder


class IntentExtractor(Protocol):
    """Protocol for intent extractors used by the agent loop."""

    def extract_intent(self, prompt: str, available_tasks: list[str]) -> IntentExtraction:
        """Extract a structured intent or clarification request."""
        ...


class StateHydrationProvider(Protocol):
    """Protocol for memory-backed state hydration."""

    async def hydrate_state(
        self,
        required_fluents: list[str],
        entities: list[str],
    ) -> dict[str, Any]:
        """Return latest known state facts relevant to a request."""
        ...


@dataclass(frozen=True)
class AgentDomainContext:
    """Static domain and problem context for a single agent run."""

    domain_path: str
    objects: dict[str, str]
    initial_state: list[dict[str, Any]] = field(default_factory=list)
    available_tasks: list[str] = field(default_factory=list)
    required_fluents: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgentRunResult:
    """Result returned by the top-level Kortex agent loop."""

    status: str
    trace: list[TraceEvent]
    extraction: HTNLaunchPad | ClarificationRequired | None = None
    plan: Plan | None = None
    execution: list[Any] = field(default_factory=list)
    clarification: ClarificationRequired | None = None


class KortexAgent:
    """Coordinates extraction, memory hydration, planning, execution, and trace logging."""

    def __init__(
        self,
        extractor: IntentExtractor,
        driver: ExecutionDriver | None = None,
        hydrator: StateHydrationProvider | None = None,
        memory_manager: MemoryManager | None = None,
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        """Initialize the agent with replaceable production or test dependencies."""
        self.extractor = extractor
        self.driver = driver or ExecutionDriver()
        self.hydrator = hydrator
        self.memory_manager = memory_manager
        self.trace_recorder = trace_recorder or TraceRecorder()

    async def run(self, prompt: str, context: AgentDomainContext) -> AgentRunResult:
        """Execute one request through the full Kortex control loop."""
        run_id = str(uuid4())
        self._trace(run_id, "request", "Received user request", {"prompt": prompt})

        extraction = self.extractor.extract_intent(prompt, context.available_tasks)
        self._trace(
            run_id,
            "extraction",
            "Extractor returned structured output",
            {"type": extraction.__class__.__name__, "value": extraction.model_dump()},
        )

        if isinstance(extraction, ClarificationRequired):
            self._trace(
                run_id,
                "hitl.clarification",
                "Execution paused for user clarification",
                extraction.model_dump(),
            )
            return AgentRunResult(
                status="clarification_required",
                extraction=extraction,
                clarification=extraction,
                trace=self._events_for_run(run_id),
            )

        initial_state = list(context.initial_state)
        if self.hydrator is not None:
            entities = self._extract_entities(extraction)
            hydrated = await self.hydrator.hydrate_state(context.required_fluents, entities)
            hydrated_facts = self._normalize_hydrated_state(hydrated)
            initial_state.extend(hydrated_facts)
            self._trace(
                run_id,
                "memory.hydration",
                "Hydrated planner state from memory",
                {"entities": entities, "facts": hydrated_facts},
            )

        planner = KortexPlanner("kortex_agent_run")
        bootstrapper = DomainBootstrapper(planner)
        bootstrapper.load_domain(context.domain_path)
        bootstrapper.load_problem_state(context.objects, initial_state)
        self._trace(
            run_id,
            "planning.bootstrap",
            "Loaded domain and problem state",
            {"domain_path": context.domain_path, "objects": context.objects},
        )

        self._create_goal(bootstrapper, planner, extraction)
        self._trace(
            run_id,
            "planning.goal",
            "Created deterministic planning goal",
            {"root_task_name": extraction.root_task_name, "parameters": extraction.task_parameters},
        )

        plan = planner.execute_plan()
        if plan is None:
            self._trace(run_id, "planning.impasse", "Planner returned no plan")
            return AgentRunResult(
                status="impasse",
                extraction=extraction,
                trace=self._events_for_run(run_id),
            )

        action_names = [action_instance.action.name for action_instance in plan.actions]
        self._trace(
            run_id,
            "planning.plan",
            "Planner produced executable plan",
            {"actions": action_names},
        )

        previous_trace_callback = self.driver.trace_callback
        self.driver.trace_callback = (
            lambda stage, message, payload: self._trace(run_id, stage, message, payload)
        )
        try:
            execution = self.driver.execute_plan(plan)
        finally:
            self.driver.trace_callback = previous_trace_callback

        self._trace(
            run_id,
            "execution.complete",
            "Executed plan through physical driver",
            {"results": execution},
        )

        self._record_execution_episodes(plan, execution)
        return AgentRunResult(
            status="success",
            extraction=extraction,
            plan=plan,
            execution=execution,
            trace=self._events_for_run(run_id),
        )

    def _create_goal(
        self,
        bootstrapper: DomainBootstrapper,
        planner: KortexPlanner,
        launchpad: HTNLaunchPad,
    ) -> None:
        """Create either an HTN task goal or a classical fluent goal from extraction."""
        goal = launchpad.task_parameters.get("goal")
        if isinstance(goal, dict):
            bootstrapper.create_goal(goal)
            return

        if launchpad.root_task_name in planner._htn_methods:
            args = launchpad.task_parameters.get("args", [])
            if not isinstance(args, list):
                raise TypeError("HTN task parameter 'args' must be a list when provided.")
            bootstrapper.create_goal({"task": launchpad.root_task_name, "args": args})
            return

        args = launchpad.task_parameters.get("args", [])
        if not isinstance(args, list):
            raise TypeError("Classical goal parameter 'args' must be a list when provided.")
        bootstrapper.create_goal(
            {
                "fluent": launchpad.root_task_name,
                "args": args,
                "value": launchpad.task_parameters.get("value", True),
            }
        )

    def _record_execution_episodes(self, plan: Plan, execution: list[Any]) -> None:
        """Append executed primitive steps to memory when a manager is configured."""
        if self.memory_manager is None:
            return

        for action_instance, result in zip(plan.actions, execution):
            kwargs = self._action_kwargs(action_instance)
            self.memory_manager.hook_post_execution(
                step_name=action_instance.action.name,
                input_payload=kwargs,
                execution_result=result,
            )

    def _action_kwargs(self, action_instance: Any) -> dict[str, str]:
        """Extract concrete action parameters from a UPF action instance."""
        kwargs: dict[str, str] = {}
        for param, actual_val in zip(
            action_instance.action.parameters,
            action_instance.actual_parameters,
        ):
            kwargs[param.name] = actual_val.object().name
        return kwargs

    def _extract_entities(self, launchpad: HTNLaunchPad) -> list[str]:
        """Collect object-like string parameters for memory hydration."""
        entities: list[str] = []
        for value in launchpad.task_parameters.values():
            if isinstance(value, str):
                entities.append(value)
            elif isinstance(value, list):
                entities.extend(str(item) for item in value)
            elif isinstance(value, dict):
                entities.extend(str(item) for item in value.get("args", []))
        return entities

    def _normalize_hydrated_state(self, hydrated_state: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert hydrator output into bootstrapper-compatible fact dictionaries."""
        facts: list[dict[str, Any]] = []
        for fluent, fact in hydrated_state.items():
            if isinstance(fact, dict):
                facts.append(
                    {
                        "fluent": fluent,
                        "args": fact.get("args", []),
                        "value": fact.get("value", True),
                    }
                )
            elif isinstance(fact, list):
                for item in fact:
                    facts.append(
                        {
                            "fluent": fluent,
                            "args": item.get("args", []),
                            "value": item.get("value", True),
                        }
                    )
        return facts

    def _trace(
        self,
        run_id: str,
        stage: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Record an agent lifecycle event."""
        self.trace_recorder.emit(
            run_id=run_id,
            stage=stage,
            message=message,
            payload=payload,
        )

    def _events_for_run(self, run_id: str) -> list[TraceEvent]:
        """Return trace events emitted for a specific run."""
        return [event for event in self.trace_recorder.events if event.run_id == run_id]
