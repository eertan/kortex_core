from typing import Any, Dict, List
import subprocess
import json

class SubagentEnclosure:
    """
    Encapsulates an external cognitive agent (e.g., Soar, Pi SDK, Fast Downward, etc.)
    Treats the external agent as an opaque tool/primitive action for the main UPF planner.
    Enforces the Read-Isolate-Write state boundary.
    """
    
    def __init__(self, name: str, command: List[str], state_ingress_keys: List[str], state_egress_mutations: List[str]):
        """
        Args:
            name: Name of the subagent/tool.
            command: The CLI command or execution string to invoke the external agent. 
                     (e.g., ['node', 'run_pi_agent.js'] or ['soar', '-c', 'run'])
            state_ingress_keys: Master UPF state fluents allowed to be passed INTO the subagent.
            state_egress_mutations: Master UPF state fluents the subagent is permitted to ALTER.
        """
        self.name = name
        self.command = command
        self.state_ingress_keys = state_ingress_keys
        self.state_egress_mutations = state_egress_mutations

    def execute(self, master_world_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Spawns the isolated external agent, passes the permitted state, and extracts mutations.
        """
        print(f"[SubagentEnclosure] Invoking isolated sub-cognition kernel: {self.name}")
        
        # 1. Boundary Guard 1: Strict Ingress Scope Gating
        isolated_state = {
            key: master_world_state[key] 
            for key in self.state_ingress_keys 
            if key in master_world_state
        }
        
        # 2. Invoke External Agent as a Black Box
        # We pass the state as a JSON string via stdin to the subprocess
        try:
            process = subprocess.run(
                self.command,
                input=json.dumps(isolated_state),
                text=True,
                capture_output=True,
                check=True
            )
            raw_output = process.stdout.strip()
            
            # Assume external agent returns a JSON string of state mutations
            # E.g., {"system_triage_classification": "under_ddos"}
            if not raw_output:
                return {}
            
            proposed_mutations = json.loads(raw_output)
            
        except subprocess.CalledProcessError as e:
            print(f"[SubagentEnclosure] External agent failed: {e.stderr}")
            raise RuntimeError(f"Subagent '{self.name}' impasse: {e.stderr}")
        except json.JSONDecodeError:
            print(f"[SubagentEnclosure] Failed to parse output from {self.name}. Output: {raw_output}")
            raise ValueError(f"Subagent '{self.name}' returned invalid JSON.")

        # 3. Boundary Guard 2: Strict Egress State Mutation Filtering
        safe_mutations = {}
        for key, value in proposed_mutations.items():
            if key in self.state_egress_mutations:
                safe_mutations[key] = value
            else:
                print(f"[SubagentEnclosure] WARNING: Subagent attempted to mutate unauthorized state '{key}'. Blocked.")
                
        print(f"[SubagentEnclosure] Sub-Cognition Exited. Safe mutations generated: {safe_mutations}")
        return safe_mutations
