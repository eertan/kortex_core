"""Tests for runtime intent frame construction from config."""

from __future__ import annotations

from pathlib import Path

from kortex.domain_package import DomainPackageLoader
from kortex.intent_runtime import IntentClarification, IntentFrame, IntentFrameBuilder


TRAVEL_PACKAGE = Path("scenarios/domains/travel_concierge")


def test_intent_frame_builder_clarifies_missing_required_slots() -> None:
    """Missing required intent slots should produce configured clarification text."""
    package = DomainPackageLoader().load(TRAVEL_PACKAGE)
    assert package.intents is not None
    builder = IntentFrameBuilder(package.intents)

    result = builder.build(
        "plan_trip",
        {
            "destination": "tokyo",
            "duration_days": 3,
            "travel_window": "next_month",
            "style": "relaxed",
        },
    )

    assert isinstance(result, IntentClarification)
    assert result.missing_slots == ["origin", "budget"]
    assert "What city are you departing from?" in result.question
    assert "What budget should I stay within?" in result.question


def test_intent_frame_builder_normalizes_slots_and_preferences() -> None:
    """Complete configured slots should become planner-ready normalized parameters."""
    package = DomainPackageLoader().load(TRAVEL_PACKAGE)
    assert package.intents is not None
    builder = IntentFrameBuilder(package.intents)

    result = builder.build(
        "plan_trip",
        {
            "origin": "boston",
            "destination": "tokyo",
            "duration_days": 3,
            "travel_window": "next_month",
            "budget": "$2500",
            "style": "relaxed",
        },
    )

    assert isinstance(result, IntentFrame)
    assert result.planner_binding == "plan_trip"
    assert result.normalized_parameters == {
        "origin": "boston",
        "destination": "tokyo",
        "duration_days": "duration_3_days",
        "travel_window": "next_month",
        "budget": "budget_2500",
        "style": "relaxed",
    }
    assert result.preference_tokens == [
        "style:relaxed",
        "relaxed",
        "boutique",
        "food_markets",
    ]


def test_intent_frame_builder_uses_default_slot_values() -> None:
    """Optional defaults from intent config should be applied before normalization."""
    package = DomainPackageLoader().load(TRAVEL_PACKAGE)
    assert package.intents is not None
    builder = IntentFrameBuilder(package.intents)

    result = builder.build(
        "plan_trip",
        {
            "origin": "boston",
            "destination": "tokyo",
            "duration_days": 3,
            "travel_window": "next_month",
            "budget": 2500,
        },
    )

    assert isinstance(result, IntentFrame)
    assert result.slots["style"] == "relaxed"
    assert result.normalized_parameters["style"] == "relaxed"


def test_intent_frame_builder_canonicalizes_object_like_strings() -> None:
    """Human text slot values should converge to planner object keys."""
    package = DomainPackageLoader().load(TRAVEL_PACKAGE)
    assert package.intents is not None
    builder = IntentFrameBuilder(package.intents)

    result = builder.build(
        "plan_trip",
        {
            "origin": "Boston",
            "destination": "Tokyo",
            "duration_days": "3 days",
            "travel_window": "next week",
            "budget": "2000 dollars",
        },
    )

    assert isinstance(result, IntentFrame)
    assert result.normalized_parameters == {
        "origin": "boston",
        "destination": "tokyo",
        "duration_days": "duration_3_days",
        "travel_window": "next_week",
        "budget": "budget_2000",
        "style": "relaxed",
    }
