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
        "travel_window": "in_30_days",
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
        "travel_window": "in_7_days",
        "budget": "budget_2000",
        "style": "relaxed",
    }


def test_intent_frame_builder_grounding_priority_unsupported() -> None:
    """If a provided slot fails grounding, we should prioritize clarifying it even if other required slots are missing."""
    package = DomainPackageLoader().load(TRAVEL_PACKAGE)
    assert package.intents is not None
    builder = IntentFrameBuilder(package.intents)

    # We pass an objects dict representing the domain inventory
    objects = {
        "boston": "City",
        "tokyo": "City",
        "next_month": "TravelWindow",
    }

    # User provides destination='japan' (ungrounded) and style='relaxed'
    # Required slots 'origin', 'duration_days', 'travel_window', 'budget' are all missing!
    # Under old logic, it would return a missing clarification for 'origin', 'duration_days', etc.
    # Under new logic, it must immediately clarify the unsupported 'destination' first!
    result = builder.build(
        "plan_trip",
        {
            "destination": "japan",
            "style": "relaxed",
        },
        objects=objects,
    )

    assert isinstance(result, IntentClarification)
    assert result.missing_slots == ["destination"]
    assert "I couldn't ground 'japan' to a specific city" in result.question


def test_intent_frame_builder_uses_normalization_aliases() -> None:
    """Slot values should be mapped using normalization_aliases if matched."""
    package = DomainPackageLoader().load(TRAVEL_PACKAGE)
    assert package.intents is not None

    # Add an alias to the catalog dynamically for testing
    spec = package.intents.intents["plan_trip"]
    spec.slots["destination"].normalization_aliases["japan"] = "tokyo"

    builder = IntentFrameBuilder(package.intents)

    # Now passing 'japan' as destination should map it to 'tokyo' via aliases
    result = builder.build(
        "plan_trip",
        {
            "origin": "boston",
            "destination": "japan",
            "duration_days": 3,
            "travel_window": "next_month",
            "budget": 2500,
        },
    )

    assert isinstance(result, IntentFrame)
    assert result.normalized_parameters["destination"] == "tokyo"
