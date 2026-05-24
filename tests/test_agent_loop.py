from typing import Any

import pytest

from kortex.agent import AgentDomainContext, KortexAgent
from kortex.extractor.models import ClarificationRequired, HTNLaunchPad, IntentExtraction
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

    def hook_post_execution(
        self,
        step_name: str,
        input_payload: dict[str, Any],
        execution_result: Any,
    ) -> None:
        """Capture execution episodes."""
        self.episodes.append((step_name, input_payload, execution_result))


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
    assert hydrator.calls == [(["robot_at"], ["vault"])]
    assert memory_manager.episodes == [
        (
            "agent_move",
            {"frm": "lobby", "to": "vault"},
            "agent moved from lobby to vault",
        )
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
