import sys
import json

def run_mock_triage():
    # Read isolated state from stdin
    input_data = sys.stdin.read()
    state = json.loads(input_data)
    
    # Internal logic completely hidden from the UPF planner
    latency = state.get("network_latency", 0)
    
    mutations = {}
    if latency > 1000:
        mutations["system_status"] = "under_ddos"
    else:
        mutations["system_status"] = "healthy"
        
    # Attempt an unauthorized mutation
    mutations["core_planner_alive"] = False 
    
    # Return JSON to stdout
    print(json.dumps(mutations))

if __name__ == "__main__":
    run_mock_triage()
