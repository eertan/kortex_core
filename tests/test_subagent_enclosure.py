import sys
from kortex.subagents.enclosure import SubagentEnclosure

def test_subagent_enclosure():
    # Define the command to run the mock python script
    command = [sys.executable, "kortex/subagents/mock_external_agent.py"]
    
    # 1. Create the enclosure with strict Read/Write boundaries
    enclosure = SubagentEnclosure(
        name="network_triage",
        command=command,
        state_ingress_keys=["network_latency", "error_rate"],
        state_egress_mutations=["system_status"] # core_planner_alive is NOT allowed
    )
    
    # 2. Master planner state (simulated)
    master_state = {
        "network_latency": 1500,
        "error_rate": 0.05,
        "core_planner_alive": True, # The subagent shouldn't even see this
        "robot_location": "lobby"
    }
    
    # 3. Execute the encapsulated subagent
    safe_mutations = enclosure.execute(master_state)
    
    # 4. Verify Boundary Enforcements
    # The external script correctly diagnosed DDoS based on latency > 1000
    assert safe_mutations.get("system_status") == "under_ddos"
    
    # The external script tried to kill the planner, but the Enclosure blocked it
    assert "core_planner_alive" not in safe_mutations
