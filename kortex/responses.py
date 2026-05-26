"""Guarded response rendering for configurable Kortex agents.

The response layer turns validated runtime facts into user-facing text. It may
use static templates for policy-critical utterances or a constrained narrator
for natural phrasing, but it never grants the narrator authority to invent
execution state.
"""

from __future__ import annotations

from string import Formatter
from typing import Any, Literal, Protocol
import json
import os

from google import genai
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


class GeminiResponseNarrator:
    """Gemini-backed response narrator for highly natural, conversational turns."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.model_name = model_name or os.environ.get(
            "KORTEX_GEMINI_MODEL",
            "gemini-3.1-pro-preview",
        )
        resolved_key = (
            api_key
            or os.environ.get("GOOGLE_AI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or ""
        )
        if not resolved_key:
            self.client = None
            return
        self.client = genai.Client(api_key=resolved_key)

    def narrate(self, frame: ResponseFrame, policy: ResponsePolicy) -> str:
        """Ask Gemini to compose a natural-sounding response from facts."""
        if self.client is None:
            return f"Selected options for {frame.response_type}."

        prompt = (
            f"You are a friendly, natural AI assistant. Convert the following "
            f"structured facts into a warm, helpful, and polite response for "
            f"the user. Do not claim anything that is not in the facts, and "
            f"do not use any of the forbidden terms. Keep the tone: {policy.tone}.\n\n"
            f"Facts:\n{json.dumps(frame.facts, indent=2)}\n\n"
            f"Forbidden terms (Do not use these!): {policy.forbidden_terms}\n\n"
            f"Please write a natural response."
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text.strip()

    def narrate_elicitation(
        self,
        intent_name: str,
        missing_slots: list[str],
        slot_clarifications: dict[str, str],
    ) -> str:
        """Draft a warm, polite conversational turn asking for missing slots."""
        if self.client is None:
            return " ".join(slot_clarifications.values())

        prompt = (
            f"You are a friendly, natural travel assistant. We are in the middle of "
            f"planning a trip ('{intent_name}'), and we need the user to provide "
            f"some missing details before we can proceed.\n\n"
            f"The missing details and their contexts are:\n"
            f"{json.dumps(slot_clarifications, indent=2)}\n\n"
            f"Draft a warm, polite, and conversational response (1-2 sentences) "
            f"acknowledging the request and asking the user for these missing "
            f"details. Do NOT ask them like a list. Make it sound like a natural conversation."
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text.strip()


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
            if not self._is_claim_in_response(expected, normalized):
                return False, f"Response missed required claim '{claim}'."
        return True, None

    def _is_claim_in_response(self, expected: Any, normalized: str) -> bool:
        """Check if a claim's expected value is represented in the response."""
        expected_str = str(expected).lower()
        if expected_str in normalized:
            return True

        # Handle float representations (e.g. 1300.0 -> 1300 or 1,300)
        if isinstance(expected, (float, int)):
            val_int = int(expected)
            val_str = str(val_int)
            val_formatted = f"{val_int:,}"
            if val_str in normalized or val_formatted in normalized:
                return True
        elif isinstance(expected, str):
            try:
                val_float = float(expected)
                val_int = int(val_float)
                val_str = str(val_int)
                val_formatted = f"{val_int:,}"
                if val_str in normalized or val_formatted in normalized:
                    return True
            except ValueError:
                pass

        # Handle word representations for single digit numbers
        digit_words = {
            1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
            6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"
        }
        try:
            val_int = int(float(str(expected)))
            if val_int in digit_words and digit_words[val_int] in normalized:
                return True
        except (ValueError, TypeError):
            pass

        return False

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
