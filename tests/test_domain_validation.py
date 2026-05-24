import pytest

from kortex.config.bootstrapper import DomainBootstrapper
from kortex.config.validation import DomainManifestError
from kortex.plugins.registry import registry
from kortex.spine.planner import KortexPlanner


def test_bootstrapper_rejects_unknown_fluent_type(tmp_path):
    domain_file = tmp_path / "domain.yaml"
    domain_file.write_text(
        """
domain_name: invalid_domain
types:
  - Location
fluents:
  robot_at:
    signature: { loc: MissingType }
actions: []
"""
    )
    bootstrapper = DomainBootstrapper(KortexPlanner("invalid_domain"))

    with pytest.raises(DomainManifestError) as exc:
        bootstrapper.load_domain(str(domain_file))

    assert "unknown type 'MissingType'" in str(exc.value)


def test_bootstrapper_rejects_unknown_method_action(tmp_path):
    domain_file = tmp_path / "domain.yaml"
    domain_file.write_text(
        """
domain_name: invalid_domain
types:
  - Location
fluents: {}
actions: []
htn_methods:
  - name: m_invalid
    target_task: invalid_task
    ordered_subtasks:
      - ["missing_action"]
"""
    )
    bootstrapper = DomainBootstrapper(KortexPlanner("invalid_domain"))

    with pytest.raises(DomainManifestError) as exc:
        bootstrapper.load_domain(str(domain_file))

    assert "unknown action 'missing_action'" in str(exc.value)


def test_bootstrapper_rejects_registered_plugin_signature_mismatch(tmp_path):
    @registry.register_action("validation_signature_action")
    def validation_signature_action(unexpected: str) -> str:
        """Plugin intentionally missing the manifest-declared parameter."""
        return unexpected

    domain_file = tmp_path / "domain.yaml"
    domain_file.write_text(
        """
domain_name: invalid_domain
types:
  - Location
fluents: {}
actions:
  - name: validation_signature_action
    parameters: { loc: Location }
    preconditions: []
    effects: []
"""
    )
    bootstrapper = DomainBootstrapper(KortexPlanner("invalid_domain"))

    with pytest.raises(DomainManifestError) as exc:
        bootstrapper.load_domain(str(domain_file))

    assert "missing parameters" in str(exc.value)
    assert "loc" in str(exc.value)
