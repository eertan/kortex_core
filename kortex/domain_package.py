"""Domain package loading for configurable Kortex agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from kortex.optimization import OptimizationPolicy
from kortex.responses import ResponsePolicy


class IntentSlotSpec(BaseModel):
    """Config for one user-facing intent slot."""

    slot_type: str
    required: bool = False
    default: Any | None = None
    clarification: str | None = None
    values: list[str] = Field(default_factory=list)
    normalization_aliases: dict[str, str] = Field(default_factory=dict)
    normalize_to_object: str | None = None


class IntentSpec(BaseModel):
    """Config for one user-facing intent."""

    description: str | None = None
    examples: list[str] = Field(default_factory=list)
    slots: dict[str, IntentSlotSpec] = Field(default_factory=dict)
    planner_binding: str
    preference_tokens: list[dict[str, str]] = Field(default_factory=list)


class DomainScopeSpec(BaseModel):
    """Config for domain scope and refusal behavior."""

    allowed_topics: list[str] = Field(default_factory=list)
    refusal_response: str = "out_of_domain"


class IntentCatalog(BaseModel):
    """Config file containing domain interaction intents."""

    domain: str
    scope: DomainScopeSpec = Field(default_factory=DomainScopeSpec)
    intents: dict[str, IntentSpec] = Field(default_factory=dict)


class DecisionSpec(BaseModel):
    """Config wrapper for one optimizer decision policy."""

    candidate_sources: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    policy: OptimizationPolicy
    output_fact: dict[str, Any] | None = None


class DecisionCatalog(BaseModel):
    """Config file containing optimizer decision policies."""

    domain: str
    decisions: dict[str, DecisionSpec] = Field(default_factory=dict)


class ResponseCatalog(BaseModel):
    """Config file containing response rendering policies."""

    domain: str
    responses: dict[str, ResponsePolicy] = Field(default_factory=dict)


class DomainPackage(BaseModel):
    """Loaded domain package paths and typed auxiliary config."""

    package_path: Path
    domain_path: Path
    intents: IntentCatalog | None = None
    decisions: DecisionCatalog | None = None
    responses: ResponseCatalog | None = None

    model_config = {"arbitrary_types_allowed": True}


class DomainPackageLoader:
    """Load a Kortex domain package directory."""

    def load(self, package_path: str | Path) -> DomainPackage:
        """Load domain, intent, decision, and response config from a package."""
        root = Path(package_path)
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Domain package directory not found: {root}")

        domain_path = root / "domain.yaml"
        if not domain_path.exists():
            raise FileNotFoundError(f"Domain package is missing domain.yaml: {root}")

        return DomainPackage(
            package_path=root,
            domain_path=domain_path,
            intents=self._load_optional(root / "intents.yaml", IntentCatalog),
            decisions=self._load_optional(root / "decisions.yaml", DecisionCatalog),
            responses=self._load_optional(root / "responses.yaml", ResponseCatalog),
        )

    def _load_optional(
        self,
        path: Path,
        model_type: type[BaseModel],
    ) -> BaseModel | None:
        """Load one optional YAML config into a Pydantic model."""
        if not path.exists():
            return None
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return model_type.model_validate(data)
