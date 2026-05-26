from typing import Any

import pytest

from kortex.agent import AgentDomainContext, KortexAgent
from kortex.extractor.models import ClarificationRequired, HTNLaunchPad, IntentExtraction
from kortex.memory.records import MemoryRecord, MemoryType, ValidatedTracePayload
from kortex.plugins.registry import PluginRegistry
from kortex.spine.driver import ExecutionDriver

agent_loop_registry = PluginRegistry()

AGENT_DOMAIN = """
domain_name: "agent_loop_domain"
types:
  - Location
fluents:
  robot_at:
    signature: { loc: Location }
actions:
  - name: agent_move
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
htn_methods:
  - name: m_agent_deliver
    target_task: agent_deliver
    parameters: { origin: Location, destination: Location }
    ordered_subtasks:
      - ["agent_move", "origin", "destination"]
intent_bindings:
  relocate_robot:
    type: goals
    required_parameters: [destination]
    goals:
      - fluent: robot_at
        args: [destination]
        value: true
  deliver_robot:
    type: task
    task: agent_deliver
    required_parameters: [destination, origin]
    args: [origin, destination]
"""

COMPETING_METHOD_DOMAIN = """
domain_name: "competing_method_domain"
types:
  - City
fluents:
  route_prepared:
    signature: { origin: City, destination: City }
actions:
  - name: fast_route
    parameters: { origin: City, destination: City }
    preconditions: []
    effects:
      - fluent: route_prepared
        args: [origin, destination]
  - name: relaxed_route
    parameters: { origin: City, destination: City }
    preconditions: []
    effects:
      - fluent: route_prepared
        args: [origin, destination]
htn_methods:
  - name: m_fast_route
    target_task: arrange_route
    parameters: { origin: City, destination: City }
    preference_matches: [fast]
    ordered_subtasks:
      - ["fast_route", "origin", "destination"]
  - name: m_relaxed_route
    target_task: arrange_route
    parameters: { origin: City, destination: City }
    preference_matches: [relaxed, style:relaxed]
    ordered_subtasks:
      - ["relaxed_route", "origin", "destination"]
intent_bindings:
  plan_route:
    type: task
    task: arrange_route
    required_parameters: [origin, destination]
    args: [origin, destination]
"""

UNORDERED_SUBTASK_DOMAIN = """
domain_name: "unordered_subtask_domain"
types:
  - City
fluents:
  flights_found:
    signature: { origin: City, destination: City }
  hotels_found:
    signature: { destination: City }
  itinerary_ready:
    signature: { destination: City }
actions:
  - name: assemble_trip
    parameters: { origin: City, destination: City }
    preconditions:
      - fluent: flights_found
        args: [origin, destination]
      - fluent: hotels_found
        args: [destination]
    effects:
      - fluent: itinerary_ready
        args: [destination]
  - name: search_hotels
    parameters: { destination: City }
    preconditions: []
    effects:
      - fluent: hotels_found
        args: [destination]
  - name: search_flights
    parameters: { origin: City, destination: City }
    preconditions: []
    effects:
      - fluent: flights_found
        args: [origin, destination]
htn_methods:
  - name: m_plan_trip_unordered
    target_task: plan_trip
    parameters: { origin: City, destination: City }
    subtasks:
      - ["assemble_trip", "origin", "destination"]
      - ["search_hotels", "destination"]
      - ["search_flights", "origin", "destination"]
intent_bindings:
  plan_travel:
    type: task
    task: plan_trip
    required_parameters: [origin, destination]
    args: [origin, destination]
"""


@agent_loop_registry.register_action("agent_move")
def agent_move(frm: str, to: str) -> str:
    """Test plugin for agent-loop planning and execution."""
    return f"agent moved from {frm} to {to}"


@agent_loop_registry.register_action("fast_route")
def fast_route(origin: str, destination: str) -> str:
    """Test plugin for a fast route candidate."""
    return f"fast route from {origin} to {destination}"


@agent_loop_registry.register_action("relaxed_route")
def relaxed_route(origin: str, destination: str) -> str:
    """Test plugin for a relaxed route candidate."""
    return f"relaxed route from {origin} to {destination}"


@agent_loop_registry.register_action("search_flights")
def search_flights(origin: str, destination: str) -> str:
    """Test plugin for flight search."""
    return f"searched flights from {origin} to {destination}"


@agent_loop_registry.register_action("search_hotels")
def search_hotels(destination: str) -> str:
    """Test plugin for hotel search."""
    return f"searched hotels in {destination}"


@agent_loop_registry.register_action("assemble_trip")
def assemble_trip(origin: str, destination: str) -> str:
    """Test plugin for itinerary assembly."""
    return f"assembled trip for {destination}"


class FakeExtractor:
    """Deterministic extractor test double."""

    def __init__(self, extraction: IntentExtraction) -> None:
        """Store the extraction object to return."""
        self.extraction = extraction

    def extract_intent(self, prompt: str, available_tasks: list[str]) -> IntentExtraction:
        """Return the preconfigured extraction."""
        return self.extraction


class FakeHydrator:
    """Memory hydrator test double."""

    def __init__(self) -> None:
        """Initialize call tracking."""
        self.calls: list[tuple[list[str], list[str]]] = []

    async def hydrate_state(
        self,
        required_fluents: list[str],
        entities: list[str],
    ) -> dict[str, Any]:
        """Return the remembered robot location."""
        self.calls.append((required_fluents, entities))
        return {"robot_at": {"args": ["lobby"], "value": True}}


class FakeMemoryManager:
    """Memory manager test double for execution writeback."""

    def __init__(self) -> None:
        """Initialize captured episodes."""
        self.episodes: list[tuple[str, dict[str, Any], Any]] = []
        self.records: list[MemoryRecord] = []

    def hook_post_execution(
        self,
        step_name: str,
        input_payload: dict[str, Any],
        execution_result: Any,
    ) -> None:
        """Capture execution episodes."""
        self.episodes.append((step_name, input_payload, execution_result))

    def hook_memory_record(self, record: MemoryRecord) -> None:
        """Capture typed memory records."""
        self.records.append(record)


@pytest.mark.asyncio
async def test_agent_loop_pauses_for_clarification(tmp_path):
    domain_file = tmp_path / "domain.yaml"
    domain_file.write_text(AGENT_DOMAIN)
    clarification = ClarificationRequired(
        question="Should the churn model target households or individuals?",
        reason="The target entity type is ambiguous.",
    )

    agent = KortexAgent(
        extractor=FakeExtractor(clarification),
        driver=ExecutionDriver(interactive=False, registry=agent_loop_registry),
        registry=agent_loop_registry,
    )
    context = AgentDomainContext(
        domain_path=str(domain_file),
        objects={"lobby": "Location", "vault": "Location"},
        available_tasks=["build_churn_model"],
    )

    result = await agent.run("Build a churn model.", context)

    assert result.status == "clarification_required"
    assert result.clarification == clarification
    assert result.working_memory is not None
    assert result.working_memory.pending_clarifications == [clarification.model_dump()]
    assert len(result.working_memory.trace_event_ids) == 3
    assert [event.stage for event in result.trace] == [
        "request",
        "extraction",
        "hitl.clarification",
    ]


@pytest.mark.asyncio
async def test_agent_loop_hydrates_plans_executes_traces_and_writes_memory(tmp_path):
    domain_file = tmp_path / "domain.yaml"
    domain_file.write_text(AGENT_DOMAIN)
    extractor = FakeExtractor(
        HTNLaunchPad(
            root_task_name="robot_at",
            task_parameters={"args": ["vault"]},
        )
    )
    hydrator = FakeHydrator()
    memory_manager = FakeMemoryManager()
    agent = KortexAgent(
        extractor=extractor,
        hydrator=hydrator,
        memory_manager=memory_manager,
        driver=ExecutionDriver(interactive=False, registry=agent_loop_registry),
        registry=agent_loop_registry,
    )
    context = AgentDomainContext(
        domain_path=str(domain_file),
        objects={"lobby": "Location", "vault": "Location"},
        available_tasks=["robot_at"],
        required_fluents=["robot_at"],
    )

    result = await agent.run("Move the robot to the vault.", context)

    assert result.status == "success"
    assert result.execution == ["agent moved from lobby to vault"]
    assert result.working_memory is not None
    assert result.working_memory.active_task == "robot_at"
    assert result.working_memory.active_entities == ["vault"]
    assert result.working_memory.planner_tier == "classical"
    assert result.working_memory.to_bootstrapper_initial_state() == [
        {"fluent": "robot_at", "args": ["lobby"], "value": False},
        {"fluent": "robot_at", "args": ["vault"], "value": True},
    ]
    assert len(result.working_memory.retrieved_memory_records) == 3
    assert len(result.working_memory.trace_event_ids) == len(result.trace)
    assert hydrator.calls == [(["robot_at"], ["vault"])]
    assert memory_manager.episodes == [
        (
            "agent_move",
            {"frm": "lobby", "to": "vault"},
            "agent moved from lobby to vault",
        )
    ]
    assert len(memory_manager.records) == 1
    trace_record = memory_manager.records[0]
    assert trace_record.memory_type == MemoryType.VALIDATED_TRACE
    assert isinstance(trace_record.payload, ValidatedTracePayload)
    assert trace_record.payload.root_task == "robot_at"
    assert trace_record.payload.planner_tier == "classical"
    assert trace_record.payload.primitive_actions == [
        {
            "action": "agent_move",
            "parameters": {"frm": "lobby", "to": "vault"},
            "result": "agent moved from lobby to vault",
        }
    ]
    assert trace_record.payload.validation_passed is True
    assert [fact.model_dump() for fact in trace_record.payload.final_facts] == [
        {"payload_type": "planner_fact", "fluent": "robot_at", "args": ["lobby"], "value": False},
        {"payload_type": "planner_fact", "fluent": "robot_at", "args": ["vault"], "value": True},
    ]
    assert [event.stage for event in result.trace] == [
        "request",
        "extraction",
        "memory.hydration",
        "planning.bootstrap",
        "planning.goal",
        "planning.plan",
        "execution.action.prepare",
        "execution.action.success",
        "execution.complete",
    ]


@pytest.mark.asyncio
async def test_agent_loop_uses_named_intent_binding_without_ordered_args(tmp_path):
    domain_file = tmp_path / "domain.yaml"
    domain_file.write_text(AGENT_DOMAIN)
    extractor = FakeExtractor(
        HTNLaunchPad(
            root_task_name="relocate_robot",
            task_parameters={"destination": "vault"},
        )
    )
    hydrator = FakeHydrator()
    agent = KortexAgent(
        extractor=extractor,
        hydrator=hydrator,
        driver=ExecutionDriver(interactive=False, registry=agent_loop_registry),
        registry=agent_loop_registry,
    )
    context = AgentDomainContext(
        domain_path=str(domain_file),
        objects={"lobby": "Location", "vault": "Location"},
        available_tasks=["relocate_robot"],
        required_fluents=["robot_at"],
    )

    result = await agent.run("Move the robot to the vault.", context)

    assert result.status == "success"
    assert result.execution == ["agent moved from lobby to vault"]
    assert result.working_memory is not None
    assert result.working_memory.active_task == "relocate_robot"
    assert result.working_memory.current_bindings == {"destination": "vault"}
    assert result.working_memory.active_entities == ["vault"]


@pytest.mark.asyncio
async def test_agent_loop_uses_named_task_binding_without_ordered_args(tmp_path):
    domain_file = tmp_path / "domain.yaml"
    domain_file.write_text(AGENT_DOMAIN)
    extractor = FakeExtractor(
        HTNLaunchPad(
            root_task_name="deliver_robot",
            task_parameters={
                "destination": "vault",
                "origin": "lobby",
            },
        )
    )
    agent = KortexAgent(
        extractor=extractor,
        driver=ExecutionDriver(interactive=False, registry=agent_loop_registry),
        registry=agent_loop_registry,
    )
    context = AgentDomainContext(
        domain_path=str(domain_file),
        objects={"lobby": "Location", "vault": "Location"},
        initial_state=[{"fluent": "robot_at", "args": ["lobby"], "value": True}],
        available_tasks=["deliver_robot"],
    )

    result = await agent.run("Deliver the robot from the lobby to the vault.", context)

    assert result.status == "success"
    assert result.execution == ["agent moved from lobby to vault"]
    assert result.working_memory is not None
    assert result.working_memory.planner_tier == "htn"
    assert result.working_memory.current_bindings == {
        "destination": "vault",
        "origin": "lobby",
    }


@pytest.mark.asyncio
async def test_agent_loop_selects_applicable_htn_method_by_preferences(tmp_path):
    domain_file = tmp_path / "domain.yaml"
    domain_file.write_text(COMPETING_METHOD_DOMAIN)
    extractor = FakeExtractor(
        HTNLaunchPad(
            root_task_name="plan_route",
            task_parameters={
                "origin": "boston",
                "destination": "rome",
                "style": "relaxed",
            },
        )
    )
    agent = KortexAgent(
        extractor=extractor,
        driver=ExecutionDriver(interactive=False, registry=agent_loop_registry),
        registry=agent_loop_registry,
    )
    context = AgentDomainContext(
        domain_path=str(domain_file),
        objects={"boston": "City", "rome": "City"},
        available_tasks=["plan_route"],
    )

    result = await agent.run("Plan a relaxed route from Boston to Rome.", context)

    assert result.status == "success"
    assert result.execution == ["relaxed route from boston to rome"]
    assert result.working_memory is not None
    assert result.working_memory.selected_method == "m_relaxed_route"
    plan_event = next(event for event in result.trace if event.stage == "planning.plan")
    assert plan_event.payload["method_selection"]["selected_method"] == "m_relaxed_route"
    assert plan_event.payload["method_selection"]["scores"] == {
        "m_fast_route": 0.0,
        "m_relaxed_route": 2.0,
    }


@pytest.mark.asyncio
async def test_agent_loop_reports_tie_impasse_for_equal_htn_methods(tmp_path):
    domain_file = tmp_path / "domain.yaml"
    domain_file.write_text(COMPETING_METHOD_DOMAIN)
    extractor = FakeExtractor(
        HTNLaunchPad(
            root_task_name="plan_route",
            task_parameters={
                "origin": "boston",
                "destination": "rome",
            },
        )
    )
    agent = KortexAgent(
        extractor=extractor,
        driver=ExecutionDriver(interactive=False, registry=agent_loop_registry),
        registry=agent_loop_registry,
    )
    context = AgentDomainContext(
        domain_path=str(domain_file),
        objects={"boston": "City", "rome": "City"},
        available_tasks=["plan_route"],
    )

    result = await agent.run("Plan a route from Boston to Rome.", context)

    assert result.status == "tie_impasse"
    assert result.execution == []
    assert result.working_memory is not None
    assert result.working_memory.planner_tier == "tie_impasse"
    tie_event = next(event for event in result.trace if event.stage == "planning.tie_impasse")
    assert tie_event.payload == {
        "task_name": "arrange_route",
        "candidate_methods": ["m_fast_route", "m_relaxed_route"],
    }


@pytest.mark.asyncio
async def test_agent_loop_orders_unordered_htn_subtasks_with_classical_planner(tmp_path):
    domain_file = tmp_path / "domain.yaml"
    domain_file.write_text(UNORDERED_SUBTASK_DOMAIN)
    extractor = FakeExtractor(
        HTNLaunchPad(
            root_task_name="plan_travel",
            task_parameters={
                "origin": "boston",
                "destination": "tokyo",
            },
        )
    )
    agent = KortexAgent(
        extractor=extractor,
        driver=ExecutionDriver(interactive=False, registry=agent_loop_registry),
        registry=agent_loop_registry,
    )
    context = AgentDomainContext(
        domain_path=str(domain_file),
        objects={"boston": "City", "tokyo": "City"},
        available_tasks=["plan_travel"],
    )

    result = await agent.run("Plan a three day trip from Boston to Tokyo.", context)

    assert result.status == "success"
    assert result.working_memory is not None
    assert result.working_memory.selected_method == "m_plan_trip_unordered"
    assert result.execution[-1] == "assembled trip for tokyo"
    assert set(result.execution[:2]) == {
        "searched flights from boston to tokyo",
        "searched hotels in tokyo",
    }
    plan_event = next(event for event in result.trace if event.stage == "planning.plan")
    assert plan_event.payload["actions"][-1] == "assemble_trip"
