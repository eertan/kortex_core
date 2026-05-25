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


@agent_loop_registry.register_action("agent_move")
def agent_move(frm: str, to: str) -> str:
    """Test plugin for agent-loop planning and execution."""
    return f"agent moved from {frm} to {to}"


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
