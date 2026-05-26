"""Tests for config-derived LLM interaction interpretation."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from kortex.configured_interpreter import (
    ConfiguredInterpreterOutput,
    ConfiguredSlotValue,
    GeminiConfiguredTurnInterpreter,
)
from kortex.domain_package import DomainPackageLoader
from kortex.intent_runtime import IntentClarification
from kortex.memory.working import WorkingMemoryState


TRAVEL_PACKAGE = Path("scenarios/domains/travel_concierge")


class FakeModels:
    """Fake structured generation namespace."""

    def __init__(self, output: ConfiguredInterpreterOutput) -> None:
        """Store the structured output to return."""
        self.output = output
        self.calls: list[dict[str, object]] = []

    def generate_content(
        self,
        *,
        model: str,
        contents: list[str],
        response_model: type[BaseModel],
    ) -> BaseModel:
        """Return the configured structured output."""
        self.calls.append(
            {
                "model": model,
                "contents": contents,
                "response_model": response_model,
            }
        )
        return self.output


class FakeClient:
    """Fake configured LLM client."""

    def __init__(self, output: ConfiguredInterpreterOutput) -> None:
        """Initialize fake model namespace."""
        self.models = FakeModels(output)


def test_gemini_configured_interpreter_uses_config_schema_and_filters_slots() -> None:
    """LLM output should be validated against configured intent slots."""
    package = DomainPackageLoader().load(TRAVEL_PACKAGE)
    output = ConfiguredInterpreterOutput(
        turn_type="task",
        intent_name="plan_trip",
        slots=[
            ConfiguredSlotValue(slot_name="origin", value="boston"),
            ConfiguredSlotValue(slot_name="destination", value="tokyo"),
            ConfiguredSlotValue(slot_name="budget", value="2000"),
            ConfiguredSlotValue(slot_name="unknown_slot", value="discard me"),
        ],
    )
    client = FakeClient(output)
    interpreter = GeminiConfiguredTurnInterpreter(
        client=client,
        model_name="fake-model",
    )

    result = interpreter.interpret_turn(
        user_text="Plan a trip to Tokyo from Boston under 2000",
        working_memory=WorkingMemoryState(session_id="s1"),
        package=package,
        pending_clarification=None,
    )

    assert result.turn_type == "task"
    assert result.intent_name == "plan_trip"
    assert result.raw_slots == {
        "origin": "boston",
        "destination": "tokyo",
        "budget": "2000",
    }
    call = client.models.calls[0]
    assert call["model"] == "fake-model"
    assert call["response_model"] is ConfiguredInterpreterOutput
    prompt = "\n".join(call["contents"])
    assert "plan_trip" in prompt
    assert "origin" in prompt
    assert "budget" in prompt


def test_gemini_configured_interpreter_includes_pending_clarification() -> None:
    """Pending clarification context should be visible to the interpreter."""
    package = DomainPackageLoader().load(TRAVEL_PACKAGE)
    output = ConfiguredInterpreterOutput(
        turn_type="clarification_answer",
        intent_name="plan_trip",
        slots=[
            ConfiguredSlotValue(slot_name="origin", value="boston"),
            ConfiguredSlotValue(slot_name="budget", value="2000"),
        ],
    )
    client = FakeClient(output)
    interpreter = GeminiConfiguredTurnInterpreter(client=client)
    clarification = IntentClarification(
        intent_name="plan_trip",
        missing_slots=["origin", "budget"],
        question="What city and budget?",
    )

    result = interpreter.interpret_turn(
        user_text="Boston and 2000 dollars",
        working_memory=WorkingMemoryState(session_id="s1"),
        package=package,
        pending_clarification=clarification,
    )

    assert result.turn_type == "clarification_answer"
    assert result.raw_slots == {"origin": "boston", "budget": "2000"}
    runtime_context = client.models.calls[0]["contents"][1]
    assert "pending_clarification" in runtime_context
    assert "origin" in runtime_context


def test_gemini_configured_interpreter_rejects_unknown_intent() -> None:
    """Unknown intent names should not enter the planner path."""
    package = DomainPackageLoader().load(TRAVEL_PACKAGE)
    output = ConfiguredInterpreterOutput(
        turn_type="task",
        intent_name="book_restaurant",
        slots=[ConfiguredSlotValue(slot_name="destination", value="tokyo")],
    )
    interpreter = GeminiConfiguredTurnInterpreter(client=FakeClient(output))

    result = interpreter.interpret_turn(
        user_text="Book a restaurant",
        working_memory=WorkingMemoryState(session_id="s1"),
        package=package,
        pending_clarification=None,
    )

    assert result.turn_type == "conversation"
    assert result.intent_name is None
