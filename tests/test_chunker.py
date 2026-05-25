import pytest
import os
import yaml
from kortex.sandbox.chunker import IntraDomainLearner

def test_macro_operator_chunking(tmp_path):
    manifest_file = tmp_path / "domain_manifest.yaml"
    learner = IntraDomainLearner(str(manifest_file))
    
    # Simulate a Tier 2 solver finding a path to unlock the vault
    preconditions = {"robot_at_lobby": True}
    plan_actions = ["move", "unlock"]
    
    learner.chunk_successful_plan(
        failed_task_name="access_vault",
        preconditions=preconditions,
        plan_actions=plan_actions
    )
    
    assert os.path.exists(manifest_file)
    
    # Verify the YAML output
    with open(manifest_file, 'r') as f:
        data = yaml.safe_load(f)
        
    assert "htn_methods" in data
    method = data["htn_methods"][0]
    
    assert method["target_task"] == "access_vault"
    assert method["name"] == "m_compiled_access_vault_2steps"
    assert method["preconditions"]["robot_at_lobby"] == True
    
    # Check the ordered subtasks list
    assert method["ordered_subtasks"][0] == ["move"]
    assert method["ordered_subtasks"][1] == ["unlock"]


def test_macro_operator_chunking_infers_condition_contract(tmp_path):
    manifest_file = tmp_path / "domain_manifest.yaml"
    manifest_file.write_text(
        """
domain_name: test_domain
types:
  - Location
fluents:
  robot_at:
    signature: { loc: Location }
  door_unlocked:
    signature: { loc: Location }
actions:
  - name: move
    parameters: { frm: Location, to: Location }
    preconditions:
      - fluent: robot_at
        args: [frm]
    effects:
      - fluent: robot_at
        args: [frm]
        value: false
      - fluent: robot_at
        args: [to]
  - name: unlock
    parameters: { loc: Location }
    preconditions:
      - fluent: robot_at
        args: [loc]
    effects:
      - fluent: door_unlocked
        args: [loc]
""",
        encoding="utf-8",
    )
    learner = IntraDomainLearner(str(manifest_file))

    learner.chunk_successful_plan(
        failed_task_name="access_secure_vault",
        preconditions={},
        plan_actions=[["move", "frm", "to"], ["unlock", "to"]],
    )

    data = yaml.safe_load(manifest_file.read_text(encoding="utf-8"))
    method = data["htn_methods"][0]

    assert method["parameters"] == {"frm": "Location", "to": "Location"}
    assert method["preconditions"] == [
        {"fluent": "robot_at", "args": ["frm"]},
    ]
    assert method["effects"] == [
        {"fluent": "robot_at", "args": ["frm"], "value": False},
        {"fluent": "robot_at", "args": ["to"], "value": True},
        {"fluent": "door_unlocked", "args": ["to"], "value": True},
    ]
    assert method["provenance"]["source"] == "intra_domain_chunking"
