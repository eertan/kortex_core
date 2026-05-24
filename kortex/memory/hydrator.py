"""
Active memory hydration for deterministic planning.

The hydrator converts planner-consumable memory facts into the fact dictionaries
expected by the domain bootstrapper. It does not ask an LLM to infer state.
"""

from typing import Any

from kortex.memory.facts import FactStore, KuzuFactStore, MemoryFact


class StateHydrator:
    """
    Queries remembered planner facts and hydrates the initial world state.

    Graphiti/Kùzu can still produce and consolidate semantic memory elsewhere,
    but the planner receives only explicit boolean facts from this boundary.
    """

    def __init__(
        self,
        db_path: str = "./kortex_kuzu_db",
        fact_store: FactStore | None = None,
    ) -> None:
        """Initialize the hydrator with a Kùzu-backed store or injected store."""
        self.db_path = db_path
        self.fact_store = fact_store or KuzuFactStore(db_path=db_path)

    async def hydrate_state(
        self,
        required_fluents: list[str],
        entities: list[str],
    ) -> dict[str, Any]:
        """
        Return recent explicit facts for requested fluents/entities.

        Args:
            required_fluents: Planner fluents that need memory context.
            entities: Entity names extracted from the current request.

        Returns:
            A mapping from fluent name to one fact dictionary or a list of fact
            dictionaries when several facts share the same fluent.
        """
        print(f"[StateHydrator] Querying memory facts for entities: {entities}")
        facts = self.fact_store.query_facts(required_fluents, entities)
        hydrated = self._group_facts(facts)
        print(f"[StateHydrator] Hydrated Context: {hydrated}")
        return hydrated

    def _group_facts(self, facts: list[MemoryFact]) -> dict[str, Any]:
        """Group memory facts by fluent while preserving bootstrapper compatibility."""
        grouped: dict[str, list[dict[str, Any]]] = {}
        for fact in facts:
            grouped.setdefault(fact.fluent, []).append(
                {"args": fact.args, "value": fact.value}
            )

        return {
            fluent: fact_group[0] if len(fact_group) == 1 else fact_group
            for fluent, fact_group in grouped.items()
        }
