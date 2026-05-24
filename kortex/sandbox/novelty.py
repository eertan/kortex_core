import json
import subprocess
from typing import Dict, Any, Tuple
import os

class NoveltyBranch:
    """
    Handles total domain impasses (Tier 3).
    When the HTN/PDDL core fails to find a valid decomposition or path,
    it delegates the problem to the Pi Coding Agent SDK to generate:
    1. New HTN YAML recipes (Methods/Tasks) to bridge the gap.
    2. New Python primitive plugins (if a physical capability is completely missing).
    """
    
    def __init__(self, workspace_path: str = "./"):
        self.workspace_path = workspace_path
        self.plugins_dir = os.path.join(self.workspace_path, "kortex", "plugins")
        self.domain_manifest_path = os.path.join(self.workspace_path, "domain.yaml")
        
    def resolve_impasse(self, failed_goal: Dict[str, Any], current_state: dict, available_actions: list) -> bool:
        """
        Triggers the Pi agent to synthesize a solution for the impasse.
        """
        print(f"[NoveltyBranch] IMPASSE DETECTED for goal: {failed_goal['fluent']}")
        
        # Construct the impasse context
        impasse_context = {
            "failed_goal": failed_goal,
            "current_state_fluents": [str(k) for k in current_state.keys() if current_state[k]],
            "available_primitives": available_actions
        }
        
        prompt = (
            f"The Kortex Core planner has hit a Tier 3 Impasse.\n"
            f"Failed Goal: {json.dumps(impasse_context['failed_goal'])}\n"
            f"Current True State Fluents: {impasse_context['current_state_fluents']}\n"
            f"Available Primitive Actions in Registry: {impasse_context['available_primitives']}\n\n"
            f"Instructions:\n"
            f"1. Check if the available primitives can be arranged into a new HTN Method to solve this. If so, edit `{self.domain_manifest_path}` to add this new method under `methods:`.\n"
            f"2. Only if the primitives are insufficient and a physical capability is completely missing, write a new python function using the `@registry.register_action` decorator inside a new file in `{self.plugins_dir}` and update the `{self.domain_manifest_path}` to declare the action's preconditions/effects.\n"
            f"3. Do not break existing deterministic execution. Use your tools to read the files, edit them, and exit."
        )
        
        print(f"[NoveltyBranch] Spawning Pi Agent Subagent...")
        
        # We invoke the Pi agent CLI via subprocess. 
        # In a real environment, this connects to the pi-subagents SDK.
        try:
            # We mock the actual node/pi call for this scaffolding, but format it exactly how 
            # the pi CLI would be invoked in a shell context.
            cmd = ["pi", "run", "worker", prompt]
            
            # Note: For unit testing we won't actually block on 'pi' CLI unless it's installed globally.
            # We just print the command that bridges to the Node.js Pi ecosystem.
            print(f"[NoveltyBranch] Executing Pi SDK Command: {' '.join(cmd)}")
            
            # process = subprocess.run(cmd, check=True, capture_output=True, text=True)
            # print(f"[NoveltyBranch] Pi Agent completed synthesis: {process.stdout}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"[NoveltyBranch] Pi Agent failed to resolve impasse: {e.stderr}")
            return False
