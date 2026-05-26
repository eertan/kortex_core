"""Config-driven travel interaction smoke tests."""

from __future__ import annotations

from pathlib import Path

from kortex.domain_package import DomainPackageLoader
from kortex.intent_runtime import IntentClarification, IntentFrame, IntentFrameBuilder
from kortex.responses import ResponseFrame, ResponseRenderer


TRAVEL_PACKAGE = Path("scenarios/domains/travel_concierge")


def test_travel_interaction_config_handles_scope_and_clarification() -> None:
    """Travel config should support early interaction decisions before planning."""
    package = DomainPackageLoader().load(TRAVEL_PACKAGE)
    assert package.intents is not None
    assert package.responses is not None
    builder = IntentFrameBuilder(package.intents)

    assert builder.in_scope("Can you help with flights and hotels?") is True
    assert builder.in_scope("Can you analyze Q4 churn?") is False

    refusal = ResponseRenderer().render(
        ResponseFrame(response_type="out_of_domain"),
        package.responses.responses["out_of_domain"],
    )
    assert "trip planning" in refusal.text

    clarification = builder.build(
        "plan_trip",
        {
            "destination": "tokyo",
            "duration_days": 3,
            "travel_window": "next_month",
        },
    )
    assert isinstance(clarification, IntentClarification)
    assert clarification.missing_slots == ["origin", "budget"]

    frame = builder.build(
        "plan_trip",
        {
            "origin": "boston",
            "destination": "tokyo",
            "duration_days": 3,
            "travel_window": "next_month",
            "budget": 2500,
        },
    )
    assert isinstance(frame, IntentFrame)
    assert frame.normalized_parameters["budget"] == "budget_2500"
