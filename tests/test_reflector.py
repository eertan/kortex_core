import pytest
import os
import yaml
from unittest.mock import MagicMock
from kortex.memory.reflector import SleepReflector, SynthesizedMetaTask

def test_sleep_reflector_synergy_detection(tmp_path):
    manifest_file = tmp_path / "domain.yaml"
    reflector = SleepReflector(str(manifest_file))
    
    # Trace 1: Fix Database Issue
    trace_a = ["ssh_login", "check_disk_space", "clear_cache", "reindex_table"]
    # Trace 2: Optimize Web Server
    trace_b = ["check_disk_space", "clear_cache", "restart_nginx"]
    # Trace 3: Weekly Maintenance
    trace_c = ["download_logs", "check_disk_space", "clear_cache"]
    
    # Internal logic should find ["check_disk_space", "clear_cache"]
    synergy = reflector._find_common_subsequence([trace_a, trace_b, trace_c])
    assert synergy == ["check_disk_space", "clear_cache"]

def test_sleep_reflector_manifest_injection(tmp_path):
    manifest_file = tmp_path / "domain.yaml"
    
    # Mock LLM API key so it doesn't actually call out
    os.environ["GEMINI_API_KEY"] = "fake-key"
    reflector = SleepReflector(str(manifest_file))
    
    # Mock the LLM returning a structured Pydantic object
    mock_response = SynthesizedMetaTask(
        meta_task_name="optimize_local_storage",
        preconditions={"server_reachable": True},
        ordered_subtasks=[["check_disk_space"], ["clear_cache"]]
    )
    
    reflector.client.models.generate_content = MagicMock(return_value=mock_response)
    
    trace_a = ["check_disk_space", "clear_cache", "reindex"]
    trace_b = ["check_disk_space", "clear_cache", "restart"]
    
    success = reflector.reflect_and_synthesize([trace_a, trace_b])
    
    assert success == True
    assert os.path.exists(str(manifest_file))
    
    with open(manifest_file, 'r') as f:
        data = yaml.safe_load(f)
        
    methods = data.get("htn_methods", [])
    assert len(methods) == 1
    assert methods[0]["target_task"] == "optimize_local_storage"
    assert methods[0]["ordered_subtasks"] == [["check_disk_space"], ["clear_cache"]]
