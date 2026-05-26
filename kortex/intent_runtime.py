"""Runtime helpers for config-defined interaction intents."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from kortex.domain_package import IntentCatalog, IntentSpec


class IntentFrame(BaseModel):
    """Canonical validated intent frame passed toward planner bindings."""

    intent_name: str
    planner_binding: str
    slots: dict[str, Any] = Field(default_factory=dict)
    normalized_parameters: dict[str, Any] = Field(default_factory=dict)
    preference_tokens: list[str] = Field(default_factory=list)


class IntentClarification(BaseModel):
    """Clarification request generated from missing configured slots."""

    intent_name: str
    missing_slots: list[str]
    question: str


class IntentFrameBuilder:
    """Build canonical intent frames from configured intent catalogs."""

    def __init__(self, catalog: IntentCatalog) -> None:
        """Initialize the builder with a loaded intent catalog."""
        self.catalog = catalog

    def build(
        self,
        intent_name: str,
        raw_slots: dict[str, Any],
    ) -> IntentFrame | IntentClarification:
        """Build a canonical intent frame or return a clarification request."""
        if intent_name not in self.catalog.intents:
            raise KeyError(f"Unknown intent '{intent_name}'.")

        spec = self.catalog.intents[intent_name]
        slots = self._apply_defaults(spec, raw_slots)
        missing = [
            slot_name
            for slot_name, slot_spec in spec.slots.items()
            if slot_spec.required and self._is_missing(slots.get(slot_name))
        ]
        if missing:
            return IntentClarification(
                intent_name=intent_name,
                missing_slots=missing,
                question=self._clarification_question(spec, missing),
            )

        normalized = {
            slot_name: self._normalize_slot_value(
                slot_name,
                slots[slot_name],
                spec,
            )
            for slot_name in spec.slots
            if slot_name in slots
        }
        return IntentFrame(
            intent_name=intent_name,
            planner_binding=spec.planner_binding,
            slots=slots,
            normalized_parameters=normalized,
            preference_tokens=self._preference_tokens(spec, slots),
        )

    def in_scope(self, user_text: str) -> bool:
        """Return whether text appears to match the configured domain scope."""
        normalized = user_text.lower()
        return any(
            topic.lower() in normalized
            for topic in self.catalog.scope.allowed_topics
        )

    def _apply_defaults(
        self,
        spec: IntentSpec,
        raw_slots: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply configured default slot values."""
        slots = dict(raw_slots)
        for slot_name, slot_spec in spec.slots.items():
            if slot_name not in slots and slot_spec.default is not None:
                slots[slot_name] = slot_spec.default
        return slots

    def _clarification_question(
        self,
        spec: IntentSpec,
        missing: list[str],
    ) -> str:
        """Return a configured clarification question for missing slots."""
        questions = [
            spec.slots[slot_name].clarification
            for slot_name in missing
            if spec.slots[slot_name].clarification
        ]
        if questions:
            return " ".join(str(question) for question in questions)
        return f"I need {', '.join(missing)} before I can continue."

    def _normalize_slot_value(
        self,
        slot_name: str,
        value: Any,
        spec: IntentSpec,
    ) -> Any:
        """Normalize one slot into a planner-facing value."""
        slot_spec = spec.slots[slot_name]
        if slot_spec.normalize_to_object is None:
            if isinstance(value, str):
                return self._canonical_object_string(value)
            return value
        normalized = slot_spec.normalize_to_object
        replacement_value = (
            self._numeric_string(value)
            if slot_spec.slot_type in {"integer", "money"}
            else value
        )
        replacements = {
            "value": replacement_value,
            "amount": self._numeric_string(value),
        }
        for key, replacement in replacements.items():
            normalized = normalized.replace(f"{{{key}}}", str(replacement))
        return normalized

    def _preference_tokens(
        self,
        spec: IntentSpec,
        slots: dict[str, Any],
    ) -> list[str]:
        """Build preference tokens from intent preference config."""
        tokens: list[str] = []
        for token_spec in spec.preference_tokens:
            if "literal" in token_spec:
                tokens.append(token_spec["literal"])
                continue
            slot_name = token_spec.get("slot")
            if not slot_name or slot_name not in slots:
                continue
            value = slots[slot_name]
            token_format = token_spec.get("format", "{value}")
            tokens.append(token_format.replace("{value}", str(value)))
            tokens.append(str(value))
        return list(dict.fromkeys(tokens))

    def _is_missing(self, value: Any) -> bool:
        """Return whether a slot value is missing."""
        return value is None or value == ""

    def _numeric_string(self, value: Any) -> str:
        """Return digits from a money-like slot value."""
        text = str(value)
        digits = "".join(char for char in text if char.isdigit())
        return digits or text

    def _canonical_object_string(self, value: str) -> str:
        """Normalize free-text object-like slot values into conventional keys."""
        return value.strip().lower().replace(" ", "_").replace("-", "_")
