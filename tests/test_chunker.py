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
