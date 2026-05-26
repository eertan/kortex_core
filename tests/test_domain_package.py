"""Tests for configurable domain package loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from kortex.domain_package import DomainPackageLoader


TRAVEL_PACKAGE = Path("scenarios/domains/travel_concierge")


def test_domain_package_loader_loads_travel_config() -> None:
    """The travel domain should load as a multi-file domain package."""
    package = DomainPackageLoader().load(TRAVEL_PACKAGE)

    assert package.domain_path.name == "domain.yaml"
    assert package.intents is not None
    assert package.decisions is not None
    assert package.responses is not None
    assert package.intents.intents["plan_trip"].planner_binding == "plan_trip"
    assert package.decisions.decisions["choose_travel_bundle"].policy.objectives[0].field == (
        "total_cost"
    )
    assert package.responses.responses["optimizer_summary"].mode == "llm_narrated"


def test_domain_package_loader_requires_domain_yaml(tmp_path) -> None:
    """A package directory without domain.yaml should fail early."""
    package_dir = tmp_path / "missing_domain"
    package_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="domain.yaml"):
        DomainPackageLoader().load(package_dir)
