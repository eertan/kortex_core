"""Guarded response rendering for configurable Kortex agents.

The response layer turns validated runtime facts into user-facing text. It may
use static templates for policy-critical utterances or a constrained narrator
for natural phrasing, but it never grants the narrator authority to invent
execution state.
"""

from __future__ import annotations

from string import Formatter
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class ResponseFrame(BaseModel):
    """Validated facts and policy boundaries for one response."""

    response_type: str
    facts: dict[str, Any] = Field(default_factory=dict)
    allowed_claims: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    required_claims: list[str] = Field(default_factory=list)
    tone: str = "friendly_concise"


class ResponsePolicy(BaseModel):
    """Configurable rendering policy for one response type."""

    response_type: str
    mode: Literal["template", "llm_narrated"] = "template"
    template: str
    fallback_template: str | None = None
    forbidden_terms: list[str] = Field(default_factory=list)
    required_fields: list[str] = Field(default_factory=list)
    tone: str = "friendly_concise"


class ResponseNarrator(Protocol):
    """Protocol for optional natural-language response narrators."""

    def narrate(self, frame: ResponseFrame, policy: ResponsePolicy) -> str:
        """Return a proposed response from a constrained response frame."""
        ...


class ResponseRenderResult(BaseModel):
    """Rendered response and audit metadata."""

    text: str
    mode_used: str
    guard_reason: str | None = None


class ResponseRenderer:
    """Render responses with deterministic guards and optional narration."""

    def __init__(self, narrator: ResponseNarrator | None = None) -> None:
        """Initialize the renderer with an optional narrator."""
        self.narrator = narrator

    def render(
        self,
        frame: ResponseFrame,
        policy: ResponsePolicy,
    ) -> ResponseRenderResult:
        """Render a response from validated facts and policy."""
        self._validate_required_fields(frame, policy)
        if policy.mode == "llm_narrated" and self.narrator is not None:
            proposed = self.narrator.narrate(frame, policy)
            allowed, reason = self._guard_response(proposed, frame, policy)
            if allowed:
                return ResponseRenderResult(text=proposed, mode_used="llm_narrated")
            fallback = policy.fallback_template or policy.template
            return ResponseRenderResult(
                text=self._render_template(fallback, frame.facts),
                mode_used="template_fallback",
                guard_reason=reason,
            )

        return ResponseRenderResult(
            text=self._render_template(policy.template, frame.facts),
            mode_used="template",
        )

    def _validate_required_fields(
        self,
        frame: ResponseFrame,
        policy: ResponsePolicy,
    ) -> None:
        """Ensure fields required by the policy are available."""
        missing = [
            field_name
            for field_name in policy.required_fields
            if self._resolve_field(frame.facts, field_name) is None
        ]
        if missing:
            raise ValueError(
                f"Response policy '{policy.response_type}' is missing fields: {missing}."
            )

    def _guard_response(
        self,
        response: str,
        frame: ResponseFrame,
        policy: ResponsePolicy,
    ) -> tuple[bool, str | None]:
        """Reject narration that contains forbidden claims or misses required ones."""
        normalized = response.lower()
        for claim in [*frame.forbidden_claims, *policy.forbidden_terms]:
            if claim.lower() in normalized:
                return False, f"Response included forbidden claim '{claim}'."

        for claim in frame.required_claims:
            expected = self._resolve_field(frame.facts, claim)
            if expected is None:
                continue
            if str(expected).lower() not in normalized:
                return False, f"Response missed required claim '{claim}'."
        return True, None

    def _render_template(self, template: str, facts: dict[str, Any]) -> str:
        """Render a format template with dotted-field support."""
        rendered = template
        for _literal, field_name, _format_spec, _conversion in Formatter().parse(template):
            if not field_name:
                continue
            value = self._resolve_field(facts, field_name)
            rendered = rendered.replace(f"{{{field_name}}}", str(value))
        return rendered

    def _resolve_field(self, facts: dict[str, Any], field_name: str) -> Any:
        """Resolve dotted field names from nested dictionaries."""
        current: Any = facts
        for part in field_name.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
                continue
            return None
        return current
