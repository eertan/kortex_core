import yaml
from typing import Sequence
import os

class IntraDomainLearner:
    """
    Implements Macro-Operator Chunking (Tier 2 Learning).
    When the PDDL state-space solver bridges a gap using primitive actions,
    this module extracts that trace and compiles it into a static HTN Method.
    This bypasses the LLM entirely and speeds up future execution via symbolic compilation.
    """
    def __init__(self, manifest_path: str = "domain_manifest.yaml"):
        self.manifest_path = manifest_path

    def chunk_successful_plan(
        self,
        failed_task_name: str,
        preconditions: dict,
        plan_actions: Sequence[str | Sequence[str]],
    ) -> None:
        """
        Takes a sequence of successful primitive actions and writes them as an HTN method.
        """
        print(f"[IntraDomainLearner] Extracting macro-operator for task '{failed_task_name}'")
        
        # 1. Format the action trace into a clean subtask sequence
        subtask_list = []
        for action_spec in plan_actions:
            if isinstance(action_spec, str):
                subtask_list.append([action_spec])
            else:
                subtask_list.append([str(part) for part in action_spec])
            
        # 2. Structure the new HTN Method "Chunk"
        new_method = {
            "name": f"m_compiled_{failed_task_name}_{len(subtask_list)}steps",
            "target_task": failed_task_name,
            "preconditions": preconditions,
            "ordered_subtasks": subtask_list
        }
        
        # 3. Append the compiled reflex back to the declarative file
        if not os.path.exists(self.manifest_path):
            print(f"[IntraDomainLearner] Manifest {self.manifest_path} not found. Creating new.")
            manifest = {"htn_methods": []}
        else:
            with open(self.manifest_path, 'r') as f:
                manifest = yaml.safe_load(f) or {}
                
        if "htn_methods" not in manifest:
            manifest["htn_methods"] = []
            
        manifest["htn_methods"].append(new_method)
        
        with open(self.manifest_path, 'w') as f:
            yaml.dump(manifest, f, sort_keys=False)
            
        print(f"[IntraDomainLearner] Macro-Operator Chunked: Compiled '{failed_task_name}' into static HTN method.")
