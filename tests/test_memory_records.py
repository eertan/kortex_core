import pytest
from pydantic import ValidationError

from kortex.memory.records import (
    ConversationMemoryPayload,
    MemoryLifecycleState,
    MemoryRecord,
    MemoryScope,
    MemorySource,
    MemoryType,
    PlannerFactPayload,
    ProceduralSkillPayload,
)
from kortex.memory.adapters import (
    planner_fact_record_from_dict,
    planner_fact_records_from_action_effects,
)
from kortex.memory.working import WorkingMemoryState


def test_memory_record_requires_payload_type_to_match_envelope() -> None:
    with pytest.raises(ValidationError, match="does not match"):
        MemoryRecord(
            memory_type=MemoryType.CONVERSATION,
            scope=MemoryScope.SESSION,
            source=MemorySource(system="test"),
            payload=PlannerFactPayload(fluent="robot_at", args=["lobby"]),
        )


def test_validated_planner_fact_hydrates_working_memory() -> None:
    record = MemoryRecord(
        memory_type=MemoryType.PLANNER_FACT,
        scope=MemoryScope.USER,
        subject_ids=["user-1", "lobby"],
        source=MemorySource(system="fact_store", reference="fact-1"),
        lifecycle_state=MemoryLifecycleState.VALIDATED,
        payload=PlannerFactPayload(fluent="robot_at", args=["lobby"], value=True),
    )
    working_memory = WorkingMemoryState(session_id="session-1", user_id="user-1")

    working_memory.hydrate_planner_fact(record)

    assert working_memory.current_facts == [
        PlannerFactPayload(fluent="robot_at", args=["lobby"], value=True)
    ]
    assert working_memory.retrieved_memory_records == [record.record_id]
    assert working_memory.to_bootstrapper_initial_state() == [
        {"fluent": "robot_at", "args": ["lobby"], "value": True}
    ]


def test_draft_planner_fact_cannot_hydrate_working_memory() -> None:
    record = MemoryRecord(
        memory_type=MemoryType.PLANNER_FACT,
        scope=MemoryScope.USER,
        source=MemorySource(system="conversation_extractor"),
        lifecycle_state=MemoryLifecycleState.DRAFT,
        payload=PlannerFactPayload(fluent="door_unlocked", args=["vault"]),
    )
    working_memory = WorkingMemoryState(session_id="session-1")

    with pytest.raises(ValueError, match="not validated or promoted"):
        working_memory.hydrate_planner_fact(record)


def test_non_fact_memory_can_be_referenced_without_hydrating_planner_truth() -> None:
    record = MemoryRecord(
        memory_type=MemoryType.CONVERSATION,
        scope=MemoryScope.SESSION,
        source=MemorySource(system="chat"),
        payload=ConversationMemoryPayload(
            role="user",
            content="When I say vault, I mean the secure archive room.",
        ),
    )
    working_memory = WorkingMemoryState(session_id="session-1")

    working_memory.remember_retrieved_record(record)

    assert working_memory.retrieved_memory_records == [record.record_id]
    assert working_memory.current_facts == []


def test_procedural_skill_payload_uses_same_record_envelope() -> None:
    record = MemoryRecord(
        memory_type=MemoryType.PROCEDURAL_SKILL,
        scope=MemoryScope.DOMAIN,
        source=MemorySource(system="intra_domain_chunker", reference="trace-1"),
        lifecycle_state=MemoryLifecycleState.PROMOTED,
        confidence=0.8,
        payload=ProceduralSkillPayload(
            target_task="access_secure_vault",
            parameters={"frm": "Location", "to": "Location"},
            preconditions=[{"fluent": "robot_at", "args": ["frm"]}],
            effects=[{"fluent": "door_unlocked", "args": ["to"], "value": True}],
            ordered_subtasks=[["move", "frm", "to"], ["unlock", "to"]],
            success_count=3,
            utility=0.75,
        ),
    )

    assert record.memory_type == MemoryType.PROCEDURAL_SKILL
    assert record.payload.target_task == "access_secure_vault"
    assert record.lifecycle_state == MemoryLifecycleState.PROMOTED


def test_planner_fact_record_adapter_wraps_fact_dict() -> None:
    record = planner_fact_record_from_dict(
        {"fluent": "robot_at", "args": ["lobby"], "value": True},
        source_system="test",
        source_reference="run-1",
    )

    assert record.memory_type == MemoryType.PLANNER_FACT
    assert record.source.system == "test"
    assert record.source.reference == "run-1"
    assert record.payload == PlannerFactPayload(
        fluent="robot_at",
        args=["lobby"],
        value=True,
    )


def test_action_effect_adapter_wraps_bound_effects() -> None:
    class FakeObject:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeActual:
        def __init__(self, name: str) -> None:
            self._object = FakeObject(name)

        def object(self) -> FakeObject:
            return self._object

    class FakeParam:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeAction:
        name = "move"
        parameters = [FakeParam("frm"), FakeParam("to")]

    class FakeActionInstance:
        action = FakeAction()
        actual_parameters = [FakeActual("lobby"), FakeActual("vault")]

    action_specs = {
        "move": {
            "effects": [
                {"fluent": "robot_at", "args": ["frm"], "value": False},
                {"fluent": "robot_at", "args": ["to"], "value": True},
            ]
        }
    }

    records = planner_fact_records_from_action_effects(
        FakeActionInstance(),
        action_specs,
        source_system="execution",
    )

    assert [record.payload for record in records] == [
        PlannerFactPayload(fluent="robot_at", args=["lobby"], value=False),
        PlannerFactPayload(fluent="robot_at", args=["vault"], value=True),
    ]
