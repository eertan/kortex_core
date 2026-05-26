"""Tests for config-aware interaction sessions."""

from __future__ import annotations

from pathlib import Path

import pytest

from kortex.configured_interaction import (
    ConfiguredInteractionSession,
    ConfiguredTurnInterpretation,
)
from kortex.domain_package import DomainPackage, DomainPackageLoader
from kortex.intent_runtime import IntentClarification
from kortex.memory.records import MemoryRecord, MemoryType
from kortex.memory.working import WorkingMemoryState
from scenarios.travel_concierge import TRAVEL_DECISIONS, build_registry


TRAVEL_PACKAGE = Path("scenarios/domains/travel_concierge")
TRAVEL_OBJECTS = {
    "boston": "City",
    "tokyo": "City",
    "next_month": "TravelWindow",
    "budget_2500": "Budget",
    "duration_3_days": "TripDuration",
    "relaxed": "TravelStyle",
}


class MemorySink:
    """Captures typed memory records."""

    def __init__(self) -> None:
        """Initialize captured records."""
        self.records: list[MemoryRecord] = []

    def hook_memory_record(self, record: MemoryRecord) -> None:
        """Capture one memory record."""
        self.records.append(record)


class SequencedInterpreter:
    """Deterministic configured turn interpreter test double."""

    def __init__(self, interpretations: list[ConfiguredTurnInterpretation]) -> None:
        """Store interpretations in call order."""
        self.interpretations = interpretations
        self.calls: list[tuple[str, WorkingMemoryState, IntentClarification | None]] = []

    def interpret_turn(
        self,
        user_text: str,
        working_memory: WorkingMemoryState,
        package: DomainPackage,
        pending_clarification: IntentClarification | None,
    ) -> ConfiguredTurnInterpretation:
        """Return the next configured interpretation."""
        del package
        self.calls.append((user_text, working_memory, pending_clarification))
        return self.interpretations.pop(0)


def load_travel_package() -> DomainPackage:
    """Load the travel concierge package."""
    return DomainPackageLoader().load(TRAVEL_PACKAGE)


@pytest.mark.asyncio
async def test_configured_session_handles_conversation_without_planning() -> None:
    """Conversation-only turns should not enter the planner path."""
    package = load_travel_package()
    memory = MemorySink()
    interpreter = SequencedInterpreter(
        [
            ConfiguredTurnInterpretation(
                turn_type="conversation",
                response_text="Good morning. I can help with trip planning.",
            )
        ]
    )
    session = ConfiguredInteractionSession(
        package=package,
        objects=TRAVEL_OBJECTS,
        interpreter=interpreter,
        memory_sink=memory,
        user_id="user-1",
    )

    result = await session.handle_turn("Good morning")

    assert result.status == "conversation"
    assert result.response_text == "Good morning. I can help with trip planning."
    assert result.execution_result is None
    assert [record.memory_type for record in memory.records] == [
        MemoryType.CONVERSATION,
        MemoryType.CONVERSATION,
    ]
    assert [record.payload.role for record in memory.records] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_configured_session_refuses_out_of_domain_task() -> None:
    """Task-like requests outside configured scope should use refusal policy."""
    package = load_travel_package()
    interpreter = SequencedInterpreter(
        [
            ConfiguredTurnInterpretation(
                turn_type="task",
                intent_name="plan_trip",
                raw_slots={},
            )
        ]
    )
    session = ConfiguredInteractionSession(
        package=package,
        objects=TRAVEL_OBJECTS,
        interpreter=interpreter,
    )

    result = await session.handle_turn("Analyze Q4 customer churn")

    assert result.status == "out_of_domain"
    assert "trip planning" in result.response_text
    assert result.execution_result is None


@pytest.mark.asyncio
async def test_configured_session_resumes_clarification_into_planner_execution() -> None:
    """Clarification answers should complete the intent frame before planning."""
    TRAVEL_DECISIONS.clear()
    package = load_travel_package()
    interpreter = SequencedInterpreter(
        [
            ConfiguredTurnInterpretation(
                turn_type="task",
                intent_name="plan_trip",
                raw_slots={
                    "destination": "tokyo",
                    "duration_days": 3,
                    "travel_window": "next_month",
                    "style": "relaxed",
                },
            ),
            ConfiguredTurnInterpretation(
                turn_type="clarification_answer",
                intent_name="plan_trip",
                raw_slots={"origin": "boston", "budget": 2500},
            ),
        ]
    )
    registry = build_registry(package)
    memory = MemorySink()
    session = ConfiguredInteractionSession(
        package=package,
        objects=TRAVEL_OBJECTS,
        registry=registry,
        interpreter=interpreter,
        memory_sink=memory,
        interactive_execution=False,
    )

    first = await session.handle_turn(
        "I need trip planning for Tokyo next month for three days"
    )
    second = await session.handle_turn("Boston, and keep it under 2500 dollars")

    assert first.status == "clarification_required"
    assert first.clarification is not None
    assert first.clarification.missing_slots == ["origin", "budget"]
    assert second.status == "approval_required"
    assert second.intent_frame is not None
    assert second.intent_frame.normalized_parameters["budget"] == "budget_2500"
    assert second.execution_result is not None
    assert "search_flights" in second.execution_result.plan_actions
    assert "optimize_travel_bundle" in second.execution_result.plan_actions
    assert second.execution_result.selected_method == "m_relaxed_local_trip"
    assert second.working_memory.current_bindings["origin"] == "boston"
    assert second.execution_result.approval_request is not None
    assert second.execution_result.approval_request["action"] == "reserve_flight_hold"
    assert second.execution_result.rendered_responses
    assert "Pacific Arc 221" in second.response_text
    assert "Yanaka Atelier Stay" in second.response_text
    assert MemoryType.OPTIMIZATION_DECISION in [
        record.memory_type
        for record in memory.records
    ]
    assert TRAVEL_DECISIONS


@pytest.mark.asyncio
async def test_configured_session_approval_turn_resumes_pending_plan() -> None:
    """Approving a HITL request should resume the pending plan without replanning."""
    TRAVEL_DECISIONS.clear()
    package = load_travel_package()
    interpreter = SequencedInterpreter(
        [
            ConfiguredTurnInterpretation(
                turn_type="task",
                intent_name="plan_trip",
                raw_slots={
                    "origin": "boston",
                    "destination": "tokyo",
                    "duration_days": 3,
                    "travel_window": "next_month",
                    "budget": 2500,
                    "style": "relaxed",
                },
            )
        ]
    )
    registry = build_registry(package)
    session = ConfiguredInteractionSession(
        package=package,
        objects=TRAVEL_OBJECTS,
        registry=registry,
        interpreter=interpreter,
        interactive_execution=False,
    )

    first = await session.handle_turn("Plan a trip with flights and hotels")
    second = await session.handle_turn("yes please")
    third = await session.handle_turn("approve it")

    assert first.status == "approval_required"
    assert first.execution_result is not None
    assert first.execution_result.approval_request is not None
    assert first.execution_result.approval_request["action"] == "reserve_flight_hold"
    assert second.status == "approval_required"
    assert second.execution_result is not None
    assert second.execution_result.approval_request is not None
    assert second.execution_result.approval_request["action"] == "reserve_hotel_hold"
    assert third.status == "success"
    assert third.execution_result is not None
    assert any(
        "Placed refundable flight hold" in str(result)
        for result in third.execution_result.execution
    )
    assert any(
        "Placed refundable hotel hold" in str(result)
        for result in third.execution_result.execution
    )
    assert third.working_memory.hitl_state == {"status": "completed"}


@pytest.mark.asyncio
async def test_configured_session_denial_turn_stops_pending_plan() -> None:
    """Denying a HITL request should stop before the gated action."""
    package = load_travel_package()
    interpreter = SequencedInterpreter(
        [
            ConfiguredTurnInterpretation(
                turn_type="task",
                intent_name="plan_trip",
                raw_slots={
                    "origin": "boston",
                    "destination": "tokyo",
                    "duration_days": 3,
                    "travel_window": "next_month",
                    "budget": 2500,
                    "style": "relaxed",
                },
            )
        ]
    )
    registry = build_registry(package)
    session = ConfiguredInteractionSession(
        package=package,
        objects=TRAVEL_OBJECTS,
        registry=registry,
        interpreter=interpreter,
        interactive_execution=False,
    )

    first = await session.handle_turn("Plan a trip with flights and hotels")
    second = await session.handle_turn("no")

    assert first.status == "approval_required"
    assert second.status == "approval_denied"
    assert "I stopped before placing holds. What would you like to change" in second.response_text
    assert second.execution_result is not None
    assert not any(
        "Placed refundable flight hold" in str(result)
        for result in second.execution_result.execution
    )
    assert second.working_memory.hitl_state is not None
    assert second.working_memory.hitl_state["status"] == "denied"


@pytest.mark.asyncio
async def test_configured_session_correction_re_plans_and_re_runs_execution() -> None:
    """A correction turn during an active HITL pause should cancel the pending plan and trigger a replan."""
    package = load_travel_package()
    interpreter = SequencedInterpreter(
        [
            ConfiguredTurnInterpretation(
                turn_type="task",
                intent_name="plan_trip",
                raw_slots={
                    "origin": "boston",
                    "destination": "tokyo",
                    "duration_days": 3,
                    "travel_window": "next_month",
                    "budget": 2500,
                    "style": "relaxed",
                },
            ),
            ConfiguredTurnInterpretation(
                turn_type="correction",
                intent_name="plan_trip",
                raw_slots={"duration_days": 5},
            ),
        ]
    )
    registry = build_registry(package)
    session = ConfiguredInteractionSession(
        package=package,
        objects=TRAVEL_OBJECTS,
        registry=registry,
        interpreter=interpreter,
        interactive_execution=False,
    )

    first = await session.handle_turn("Plan a trip with flights and hotels")
    assert first.status == "approval_required"
    assert first.intent_frame is not None
    assert first.intent_frame.normalized_parameters["duration_days"] == "duration_3_days"

    second = await session.handle_turn("I changed my mind, make it 5 days please")

    assert second.status == "approval_required"
    assert second.intent_frame is not None
    assert second.intent_frame.normalized_parameters["duration_days"] == "duration_5_days"
    assert second.intent_frame.normalized_parameters["origin"] == "boston"
    assert second.intent_frame.normalized_parameters["destination"] == "tokyo"

