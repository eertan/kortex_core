"""Adapters between planner/runtime structures and memory records."""

from __future__ import annotations

from typing import Any

from kortex.memory.records import (
    MemoryLifecycleState,
    MemoryRecord,
    MemoryScope,
    MemorySource,
    MemoryType,
    PlannerFactPayload,
)


def planner_fact_record_from_dict(
    fact: dict[str, Any],
    *,
    source_system: str,
    source_reference: str | None = None,
    scope: MemoryScope = MemoryScope.SESSION,
    lifecycle_state: MemoryLifecycleState = MemoryLifecycleState.VALIDATED,
) -> MemoryRecord:
    """Wrap a bootstrapper-compatible fact dict in a planner fact memory record."""
    args = [str(arg) for arg in fact.get("args", [])]
    return MemoryRecord(
        memory_type=MemoryType.PLANNER_FACT,
        scope=scope,
        subject_ids=args,
        source=MemorySource(system=source_system, reference=source_reference),
        lifecycle_state=lifecycle_state,
        payload=PlannerFactPayload(
            fluent=fact["fluent"],
            args=args,
            value=fact.get("value", True),
        ),
    )


def planner_fact_records_from_action_effects(
    action_instance: Any,
    action_specs: dict[str, dict[str, Any]],
    *,
    source_system: str,
    source_reference: str | None = None,
) -> list[MemoryRecord]:
    """Convert a planned action instance's declared effects into fact records."""
    action_name = action_instance.action.name
    action_spec = action_specs.get(action_name)
    if action_spec is None:
        return []

    bindings = _action_bindings(action_instance)
    records: list[MemoryRecord] = []
    for effect in action_spec.get("effects", []):
        fact = {
            "fluent": effect["fluent"],
            "args": [bindings.get(str(arg), str(arg)) for arg in effect.get("args", [])],
            "value": effect.get("value", True),
        }
        records.append(
            planner_fact_record_from_dict(
                fact,
                source_system=source_system,
                source_reference=source_reference,
                lifecycle_state=MemoryLifecycleState.VALIDATED,
            )
        )
    return records


def _action_bindings(action_instance: Any) -> dict[str, str]:
    """Extract action parameter bindings from a UPF action instance."""
    bindings: dict[str, str] = {}
    for parameter, actual_value in zip(
        action_instance.action.parameters,
        action_instance.actual_parameters,
        strict=True,
    ):
        bindings[parameter.name] = actual_value.object().name
    return bindings

