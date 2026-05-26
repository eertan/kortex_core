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
        objects: dict[str, str] | None = None,
    ) -> IntentFrame | IntentClarification:
        """Build a canonical intent frame or return a clarification request."""
        if intent_name not in self.catalog.intents:
            raise KeyError(f"Unknown intent '{intent_name}'.")

        spec = self.catalog.intents[intent_name]
        slots = self._apply_defaults(spec, raw_slots)

        slot_statuses: dict[str, str] = {}
        normalized: dict[str, Any] = {}

        for slot_name, slot_spec in spec.slots.items():
            raw_val = slots.get(slot_name)
            if self._is_missing(raw_val):
                if slot_spec.required:
                    slot_statuses[slot_name] = "missing"
                else:
                    slot_statuses[slot_name] = "grounded"
                continue

            # Apply normalization aliases if provided
            val_str = str(raw_val).strip().lower()
            if val_str in slot_spec.normalization_aliases:
                raw_val = slot_spec.normalization_aliases[val_str]
                slots[slot_name] = raw_val

            norm_val = self._normalize_slot_value(slot_name, raw_val, spec)
            normalized[slot_name] = norm_val

            # Check slot validation values
            if slot_spec.values:
                if norm_val not in slot_spec.values and str(raw_val) not in slot_spec.values:
                    slot_statuses[slot_name] = "unsupported"
                    continue

            # Check grounding against planner objects if slot is not integer/money/enum
            if slot_spec.slot_type not in {"integer", "money", "enum"}:
                if objects is not None:
                    if objects.get(norm_val) == slot_spec.slot_type:
                        slot_statuses[slot_name] = "grounded"
                    else:
                        slot_statuses[slot_name] = "unsupported"
                else:
                    slot_statuses[slot_name] = "grounded"
            else:
                slot_statuses[slot_name] = "grounded"

        unsupported = [
            slot_name
            for slot_name in spec.slots
            if slot_statuses[slot_name] == "unsupported"
        ]
        if unsupported:
            return IntentClarification(
                intent_name=intent_name,
                missing_slots=unsupported,
                question=self._grounding_question_for_unsupported(
                    spec, unsupported, slots, objects
                ),
            )

        missing = [
            slot_name
            for slot_name in spec.slots
            if slot_statuses[slot_name] == "missing"
        ]
        if missing:
            return IntentClarification(
                intent_name=intent_name,
                missing_slots=missing,
                question=self._clarification_question(spec, missing),
            )

        return IntentFrame(
            intent_name=intent_name,
            planner_binding=spec.planner_binding,
            slots=slots,
            normalized_parameters=normalized,
            preference_tokens=self._preference_tokens(spec, slots),
        )

    def _grounding_question_for_unsupported(
        self,
        spec: IntentSpec,
        unsupported_slots: list[str],
        raw_slots: dict[str, Any],
        objects: dict[str, str] | None = None,
    ) -> str:
        """Build user-facing grounding clarification question for unsupported slots."""
        questions = []
        for slot_name in unsupported_slots:
            slot_spec = spec.slots[slot_name]
            raw_value = raw_slots.get(slot_name)
            slot_type = slot_spec.slot_type
            if slot_name == "destination" and slot_type == "City":
                q = (
                    f"I couldn't ground '{raw_value}' to a specific city. "
                    "What city do you want to visit?"
                )
            elif slot_name == "origin" and slot_type == "City":
                q = (
                    f"I couldn't ground '{raw_value}' to a supported departure city. "
                    "What city are you departing from?"
                )
            elif slot_spec.clarification:
                q = f"I couldn't ground '{raw_value}'. {slot_spec.clarification}"
            else:
                q = f"I couldn't ground '{raw_value}' for {slot_name}. Can you clarify?"
            questions.append(q)
        return " ".join(questions)

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
