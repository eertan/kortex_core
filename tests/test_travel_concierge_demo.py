"""Smoke tests for the travel concierge scenario package."""

from __future__ import annotations

import json

from scenarios.travel_concierge import run_travel_demo


def test_travel_concierge_demo_runs_with_approval(tmp_path) -> None:
    """Verify the travel demo runs through planning, HITL approval, and logging."""
    log_path = tmp_path / "travel_log.json"

    run_travel_demo(log_path=log_path, approval="y")

    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert len(payload) == 1
    log = payload[0]
    assert log["scenario"] == "travel_concierge"
    assert log["plan"][-1]["action"] == "finalize_trip_plan"
    reserve_flight = next(
        step
        for step in log["plan"]
        if step["action"] == "reserve_flight_hold"
    )
    assert reserve_flight["parameters"] == {
        "origin": "boston",
        "destination": "tokyo",
    }
    assert len(log["results"]) == 8
    event_stages = [event["stage"] for event in log["events"]]
    assert "interaction.intent_frame" in event_stages
    assert event_stages.index("interaction.intent_frame") < event_stages.index(
        "planning.method_candidates"
    )
    assert "planning.method_candidates" in event_stages
    assert "planning.preference_input" in event_stages
    assert "planning.method_selection" in event_stages
    assert "planning.classical_subtask_ordering" in event_stages
    assert event_stages.count("memory.option_hydration") == 2
    assert "optimization.decision" in event_stages
    assert "memory.optimization_decision" in event_stages
    assert "response.optimizer_summary" in event_stages
    assert event_stages.index("planning.classical_subtask_ordering") < event_stages.index(
        "optimization.decision"
    )
    assert event_stages.index("optimization.decision") < event_stages.index(
        "hitl.approval.required"
    )
    assert event_stages.index("response.optimizer_summary") < event_stages.index(
        "hitl.approval.required"
    )
    assert any(
        event["stage"] == "hitl.approval.granted"
        for event in log["events"]
    )
    ordering_event = next(
        event
        for event in log["events"]
        if event["stage"] == "planning.classical_subtask_ordering"
    )
    assert ordering_event["payload"]["declared_subtask_order"][0] == "finalize_trip_plan"
    assert ordering_event["payload"]["planned_action_order"][-1] == "finalize_trip_plan"
    optimization_event = next(
        event
        for event in log["events"]
        if event["stage"] == "optimization.decision"
    )
    assert optimization_event["payload"]["selected_candidate_ids"] == [
        "flight_refundable_balanced",
        "hotel_boutique_quiet",
    ]
    assert optimization_event["payload"]["selected_attributes"]["total_cost"] == 1560.0
    assert optimization_event["payload"]["selected_attributes"]["flight.name"] == (
        "Pacific Arc 221"
    )
    assert optimization_event["payload"]["selected_attributes"]["hotel.name"] == (
        "Yanaka Atelier Stay"
    )
    response_event = next(
        event
        for event in log["events"]
        if event["stage"] == "response.optimizer_summary"
    )
    assert response_event["payload"]["mode_used"] == "llm_narrated"
    assert "Pacific Arc 221" in response_event["payload"]["text"]
    assert "Yanaka Atelier Stay" in response_event["payload"]["text"]
    assert "booking confirmed" not in response_event["payload"]["text"].lower()
    option_events = [
        event
        for event in log["events"]
        if event["stage"] == "memory.option_hydration"
    ]
    assert option_events[0]["payload"]["option_count"] == 3
    assert option_events[0]["payload"]["may_hydrate_planner"] is False
    assert any(
        "m_relaxed_local_trip" in note
        for note in log["notes"]
    )
    assert any(
        "trip_plan_ready" in note
        for note in log["notes"]
    )
