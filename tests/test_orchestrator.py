import pytest
from kortex.spine.planner import KortexPlanner
from kortex.config.bootstrapper import DomainBootstrapper
from kortex.spine.orchestrator import Orchestrator
from tests.test_e2e_bootstrapper import YAML_DOMAIN
from unified_planning.shortcuts import Fluent, Object

def test_async_multi_goal_orchestration(tmp_path):
    domain_file = tmp_path / "domain.yaml"
    domain_file.write_text(YAML_DOMAIN)
    
    planner = KortexPlanner("test_spine")
    bootstrapper = DomainBootstrapper(planner)
    bootstrapper.load_domain(str(domain_file))
    
    # We have two robots and two vaults to prove concurrent execution
    objects = {
        "lobby_1": "Location", "vault_1": "Location",
        "lobby_2": "Location", "vault_2": "Location"
    }
    
    initial_state_raw = [
        {"fluent": "robot_at", "args": ["lobby_1"], "value": True},
        {"fluent": "door_unlocked", "args": ["lobby_1"], "value": True},
        {"fluent": "robot_at", "args": ["lobby_2"], "value": True},
        {"fluent": "door_unlocked", "args": ["lobby_2"], "value": True},
    ]
    bootstrapper.load_problem_state(objects, initial_state_raw)
    
    # We need to construct the internal UPF state mapping to pass to the orchestrator
    master_state = {}
    fl = bootstrapper.fluents["robot_at"]
    master_state[fl(bootstrapper.objects["lobby_1"])] = True
    master_state[fl(bootstrapper.objects["lobby_2"])] = True
    
    fl_door = bootstrapper.fluents["door_unlocked"]
    master_state[fl_door(bootstrapper.objects["lobby_1"])] = True
    master_state[fl_door(bootstrapper.objects["lobby_2"])] = True

    # Define two completely independent goals
    goals = [
        {"fluent": "robot_at", "args": ["vault_1"], "value": True},
        {"fluent": "robot_at", "args": ["vault_2"], "value": True}
    ]
    
    orchestrator = Orchestrator(bootstrapper)
    
    # Verify Independence heuristic
    assert orchestrator._are_goals_independent(goals) == True
    
    # Dispatch!
    results = orchestrator.dispatch(goals, master_state)
    
    assert results["robot_at"]["status"] == "success"
    # The output from the python plugins should be in the execution list
    execution_outputs = str(results["robot_at"]["execution"])
    assert "Robot drove from" in execution_outputs
