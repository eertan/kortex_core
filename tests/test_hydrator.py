import pytest
from kortex.memory.facts import InMemoryFactStore, KuzuFactStore, MemoryFact
from kortex.memory.hydrator import StateHydrator

@pytest.mark.asyncio
async def test_state_hydrator():
    fact_store = InMemoryFactStore(
        [
            MemoryFact(fluent="robot_at", args=["lobby"], value=True),
            MemoryFact(fluent="door_unlocked", args=["vault"], value=False),
            MemoryFact(fluent="battery_charged", args=["robot"], value=True),
        ]
    )
    hydrator = StateHydrator(fact_store=fact_store)
    
    required = ["robot_at", "door_unlocked"]
    entities = ["vault", "lobby"]
    
    state = await hydrator.hydrate_state(required, entities)
    
    assert "robot_at" in state
    assert state["robot_at"]["args"] == ["lobby"]
    
    assert "door_unlocked" in state
    assert state["door_unlocked"]["args"] == ["vault"]
    assert state["door_unlocked"]["value"] == False

    assert "battery_charged" not in state


def test_kuzu_fact_store_persists_and_queries_facts(tmp_path):
    store = KuzuFactStore(db_path=str(tmp_path / "facts_db"))
    store.upsert_fact(MemoryFact(fluent="robot_at", args=["lobby"], value=True))
    store.upsert_fact(MemoryFact(fluent="door_unlocked", args=["vault"], value=False))

    facts = store.query_facts(
        required_fluents=["robot_at", "door_unlocked"],
        entities=["vault"],
    )

    assert len(facts) == 1
    assert facts[0].fluent == "door_unlocked"
    assert facts[0].args == ["vault"]
    assert facts[0].value is False
