"""
Deterministic memory fact storage for planner state hydration.

Graphiti remains the long-term episodic/semantic layer, while this module
provides a small explicit fact interface for facts the deterministic planner can
consume directly.
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

import kuzu


@dataclass(frozen=True)
class MemoryFact:
    """A planner-consumable fact remembered by the memory subsystem."""

    fluent: str
    args: list[str]
    value: bool = True
    observed_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    fact_id: str = field(default_factory=lambda: str(uuid4()))


class FactStore(Protocol):
    """Storage boundary for planner-consumable memory facts."""

    def upsert_fact(self, fact: MemoryFact) -> None:
        """Persist or append a memory fact."""
        ...

    def query_facts(
        self,
        required_fluents: list[str],
        entities: list[str],
    ) -> list[MemoryFact]:
        """Return relevant facts for the requested fluents and entities."""
        ...


class InMemoryFactStore:
    """Deterministic in-process fact store for tests and embedded runs."""

    def __init__(self, facts: list[MemoryFact] | None = None) -> None:
        """Initialize the fact store with optional seed facts."""
        self._facts = list(facts or [])

    def upsert_fact(self, fact: MemoryFact) -> None:
        """Append a memory fact."""
        self._facts.append(fact)

    def query_facts(
        self,
        required_fluents: list[str],
        entities: list[str],
    ) -> list[MemoryFact]:
        """Return facts matching requested fluent names and entity references."""
        return _filter_facts(self._facts, required_fluents, entities)


class KuzuFactStore:
    """Kùzu-backed store for latest planner-consumable facts."""

    def __init__(self, db_path: str = "./kortex_kuzu_db") -> None:
        """Open a Kùzu database and ensure the fact schema exists."""
        self.db_path = db_path
        self.database = kuzu.Database(db_path)
        self.connection = kuzu.Connection(self.database)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create the fact node table when it does not exist."""
        self.connection.execute(
            "CREATE NODE TABLE IF NOT EXISTS Fact("
            "id STRING, "
            "fluent STRING, "
            "args STRING, "
            "value BOOL, "
            "observed_at STRING, "
            "PRIMARY KEY(id)"
            ")"
        )

    def upsert_fact(self, fact: MemoryFact) -> None:
        """Persist a planner fact in Kùzu."""
        self.connection.execute(
            "CREATE (f:Fact {"
            "id: $id, "
            "fluent: $fluent, "
            "args: $args, "
            "value: $value, "
            "observed_at: $observed_at"
            "})",
            {
                "id": fact.fact_id,
                "fluent": fact.fluent,
                "args": json.dumps(fact.args),
                "value": fact.value,
                "observed_at": fact.observed_at,
            },
        )

    def query_facts(
        self,
        required_fluents: list[str],
        entities: list[str],
    ) -> list[MemoryFact]:
        """Return Kùzu facts matching requested fluent names and entity references."""
        result = self.connection.execute(
            "MATCH (f:Fact) "
            "RETURN f.id, f.fluent, f.args, f.value, f.observed_at "
            "ORDER BY f.observed_at DESC"
        )
        facts = [
            MemoryFact(
                fact_id=row[0],
                fluent=row[1],
                args=[str(arg) for arg in json.loads(row[2])],
                value=bool(row[3]),
                observed_at=row[4],
            )
            for row in result.get_all()
        ]
        return _filter_facts(facts, required_fluents, entities)


def _filter_facts(
    facts: list[MemoryFact],
    required_fluents: list[str],
    entities: list[str],
) -> list[MemoryFact]:
    """Filter facts by fluent and optional entity overlap."""
    required = set(required_fluents)
    entity_set = set(entities)
    selected: list[MemoryFact] = []

    for fact in facts:
        if required and fact.fluent not in required:
            continue
        if entity_set and not entity_set.intersection(fact.args):
            continue
        selected.append(fact)

    return selected
