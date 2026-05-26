"""Tests for the travel interaction CLI entrypoint."""

from __future__ import annotations

import json

import pytest

from scenarios.travel_interaction_cli import (
    TravelDemoInterpreter,
    run_scripted_travel_conversation,
    run_turns,
)


@pytest.mark.asyncio
async def test_scripted_travel_interaction_approval_writes_transcript(tmp_path) -> None:
    """Scripted approval should complete the travel interaction demo."""
    log_path = tmp_path / "travel_interaction.json"

    transcript = await run_scripted_travel_conversation(
        log_path=log_path,
        approval="approve",
    )

    turns = [entry for entry in transcript if entry["type"] == "turn"]
    assert [entry["status"] for entry in turns] == [
        "conversation",
        "clarification_required",
        "approval_required",
        "approval_required",
        "success",
    ]
    assert "Pacific Arc 221" in turns[2]["assistant"]
    assert "Approve placing the refundable flight hold" in turns[2]["assistant"]
    assert "Approve placing the refundable hotel hold" in turns[3]["assistant"]
    assert turns[2]["approval_request"]["action"] == "reserve_flight_hold"
    assert turns[3]["approval_request"]["action"] == "reserve_hotel_hold"
    assert any(
        "Placed refundable hotel hold" in str(result)
        for result in turns[4]["execution"]
    )
    persisted = json.loads(log_path.read_text(encoding="utf-8"))
    assert persisted[-1]["type"] == "memory_summary"


@pytest.mark.asyncio
async def test_scripted_travel_interaction_denial_stops_before_hold(tmp_path) -> None:
    """Scripted denial should stop before running the approval-gated action."""
    log_path = tmp_path / "travel_interaction_denied.json"

    transcript = await run_scripted_travel_conversation(
        log_path=log_path,
        approval="deny",
    )

    turns = [entry for entry in transcript if entry["type"] == "turn"]
    assert [entry["status"] for entry in turns] == [
        "conversation",
        "clarification_required",
        "approval_required",
        "approval_denied",
    ]
    assert "I stopped before placing holds. What would you like to change" in turns[-1]["assistant"]
    assert not any(
        "Placed refundable flight hold" in str(result)
        for result in turns[-1]["execution"]
    )
    assert log_path.exists()


def test_travel_demo_interpreter_extracts_dense_two_turn_request() -> None:
    """Interpreter should preserve slots from natural dense demo phrasing."""
    interpreter = TravelDemoInterpreter()

    first = interpreter._travel_slots(
        "want to plan a trip to japan, i have 2000$ budget"
    )
    second = interpreter._travel_slots(
        "departing from boston, want to visit tokyo for 3 days. "
        "next week, and budget is 2000$"
    )

    assert first == {
        "destination": "japan",
        "budget": 2000,
        "style": "relaxed",
    }
    assert second == {
        "destination": "tokyo",
        "duration_days": 3,
        "travel_window": 7,
        "style": "relaxed",
        "origin": "boston",
        "budget": 2000,
    }


@pytest.mark.asyncio
async def test_dense_two_turn_request_reaches_approval() -> None:
    """A dense clarification answer should complete all missing travel slots."""
    transcript = await run_turns(
        [
            "want to plan a trip to Japan, I have 2000$ budget",
            (
                "departing from Boston, want to visit Tokyo for 3 days. "
                "Next week, and budget is 2000$"
            ),
        ]
    )

    turns = [entry for entry in transcript if entry["type"] == "turn"]
    assert [entry["status"] for entry in turns] == [
        "clarification_required",
        "approval_required",
    ]
    assert "Where are you going?" not in turns[0]["assistant"]
    assert "What budget should I stay within?" not in turns[0]["assistant"]
    assert turns[1]["intent_frame"]["normalized_parameters"]["budget"] == "budget_2000"
    assert turns[1]["intent_frame"]["normalized_parameters"]["travel_window"] == "in_7_days"


@pytest.mark.asyncio
async def test_country_destination_requires_grounding_city() -> None:
    """Country-level destinations should ask for a city instead of casting."""
    transcript = await run_turns(
        [
            "I want to travel to japan and have a budget of 2000$",
            "I am departing from New York",
            "4 daya",
            "2 days from now",
        ]
    )

    turns = [entry for entry in transcript if entry["type"] == "turn"]
    assert [entry["status"] for entry in turns] == [
        "clarification_required",
        "clarification_required",
        "clarification_required",
        "clarification_required",
    ]
    assert "specific city" in turns[0]["assistant"]
    assert "japan" in turns[0]["assistant"].lower()
    assert "Where are you going?" in turns[-1]["assistant"]


@pytest.mark.asyncio
async def test_change_of_mind_denies_pending_approval() -> None:
    """Changing mind during approval should deny the pending gated action."""
    transcript = await run_turns(
        [
            (
                "I want to visit Tokyo, traveling from boston. I have 2000$ "
                "and will stay 4 days. I want to leave in 2 days"
            ),
            "I changed my mind, want to stay 5 days",
        ]
    )

    turns = [entry for entry in transcript if entry["type"] == "turn"]
    assert turns[0]["status"] == "approval_required"
    assert turns[1]["status"] == "approval_denied"
    assert "I stopped before placing holds. What would you like to change" in turns[1]["assistant"]
