"""Tests for guarded response rendering."""

from __future__ import annotations

import pytest

from kortex.responses import (
    ResponseFrame,
    ResponsePolicy,
    ResponseRenderer,
)


class FakeNarrator:
    """Deterministic response narrator test double."""

    def __init__(self, response: str) -> None:
        """Store the response to return."""
        self.response = response

    def narrate(self, frame: ResponseFrame, policy: ResponsePolicy) -> str:
        """Return the configured response."""
        del frame, policy
        return self.response


def test_response_renderer_renders_template_with_dotted_fields() -> None:
    renderer = ResponseRenderer()
    frame = ResponseFrame(
        response_type="optimizer_summary",
        facts={
            "flight": {"name": "Pacific Arc 221"},
            "hotel": {"name": "Yanaka Atelier Stay"},
            "total_cost": 1560,
        },
    )
    policy = ResponsePolicy(
        response_type="optimizer_summary",
        mode="template",
        template="{flight.name} with {hotel.name} costs ${total_cost}.",
    )

    result = renderer.render(frame, policy)

    assert result.text == "Pacific Arc 221 with Yanaka Atelier Stay costs $1560."
    assert result.mode_used == "template"


def test_response_renderer_allows_guarded_narration() -> None:
    renderer = ResponseRenderer(
        narrator=FakeNarrator(
            "I selected Pacific Arc 221 with Yanaka Atelier Stay for $1560."
        )
    )
    frame = ResponseFrame(
        response_type="optimizer_summary",
        facts={
            "flight": {"name": "Pacific Arc 221"},
            "hotel": {"name": "Yanaka Atelier Stay"},
            "total_cost": 1560,
        },
        required_claims=["flight.name", "hotel.name"],
        forbidden_claims=["booking confirmed"],
    )
    policy = ResponsePolicy(
        response_type="optimizer_summary",
        mode="llm_narrated",
        template="{flight.name} with {hotel.name} costs ${total_cost}.",
    )

    result = renderer.render(frame, policy)

    assert result.mode_used == "llm_narrated"
    assert "Pacific Arc 221" in result.text


def test_response_renderer_falls_back_when_narration_overclaims() -> None:
    renderer = ResponseRenderer(
        narrator=FakeNarrator("Booking confirmed for Pacific Arc 221.")
    )
    frame = ResponseFrame(
        response_type="optimizer_summary",
        facts={
            "flight": {"name": "Pacific Arc 221"},
            "hotel": {"name": "Yanaka Atelier Stay"},
            "total_cost": 1560,
        },
        forbidden_claims=["booking confirmed"],
    )
    policy = ResponsePolicy(
        response_type="optimizer_summary",
        mode="llm_narrated",
        template="{flight.name} with {hotel.name} costs ${total_cost}.",
    )

    result = renderer.render(frame, policy)

    assert result.mode_used == "template_fallback"
    assert result.guard_reason == "Response included forbidden claim 'booking confirmed'."
    assert result.text == "Pacific Arc 221 with Yanaka Atelier Stay costs $1560."


def test_response_renderer_rejects_missing_required_fields() -> None:
    renderer = ResponseRenderer()
    frame = ResponseFrame(response_type="summary", facts={})
    policy = ResponsePolicy(
        response_type="summary",
        mode="template",
        template="{flight.name}",
        required_fields=["flight.name"],
    )

    with pytest.raises(ValueError, match="missing fields"):
        renderer.render(frame, policy)


def test_response_renderer_guards_budget_constraint_overclaim() -> None:
    """The renderer must fallback and reject narration if the narrator falsely claims the budget constraint is met."""
    budget_limit = 1500
    total_cost = 1560  # Over budget!

    forbidden_claims = ["booking confirmed"]
    if total_cost > budget_limit:
        forbidden_claims.append("within your budget")

    renderer = ResponseRenderer(
        narrator=FakeNarrator(
            "Pacific Arc 221 with Yanaka Atelier Stay for $1560, and it stays within your budget."
        )
    )

    frame = ResponseFrame(
        response_type="optimizer_summary",
        facts={
            "flight": {"name": "Pacific Arc 221"},
            "hotel": {"name": "Yanaka Atelier Stay"},
            "total_cost": total_cost,
        },
        forbidden_claims=forbidden_claims,
    )

    policy = ResponsePolicy(
        response_type="optimizer_summary",
        mode="llm_narrated",
        template="{flight.name} with {hotel.name} costs ${total_cost}.",
    )

    result = renderer.render(frame, policy)

    assert result.mode_used == "template_fallback"
    assert result.guard_reason == "Response included forbidden claim 'within your budget'."
    assert result.text == "Pacific Arc 221 with Yanaka Atelier Stay costs $1560."

