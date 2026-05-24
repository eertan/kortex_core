import os
import yaml
from typing import List, Dict, Any
import instructor
from google import genai
from pydantic import BaseModel, Field

class SynthesizedMetaTask(BaseModel):
    """
    The schema for a new HTN Meta-Task discovered via Sleep Reflection.
    """
    meta_task_name: str = Field(description="The semantic name of the new high-level abstract task.")
    preconditions: Dict[str, bool] = Field(description="The preconditions required to run this meta-task.")
    ordered_subtasks: List[List[str]] = Field(description="The sequence of primitive actions that define this meta-task.")

class SleepReflector:
    """
    Implements Inductive HTN Grammar Learning (Sleep-Phase Metacognition).
    Analyzes historical episodic traces to discover synergies (common sub-sequences)
    and uses the LLM to name and formalize them as new HTN Meta-Tasks.
    """
    
    def __init__(self, manifest_path: str = "domain_manifest.yaml", model_name: str = "gemini-2.5-flash"):
        self.manifest_path = manifest_path
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.model_name = model_name
        
        if self.api_key:
            base_client = genai.Client(api_key=self.api_key)
            self.client = instructor.from_genai(
                client=base_client,
                mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS
            )

    def _find_common_subsequence(self, traces: List[List[str]]) -> List[str]:
        """
        A simplified longest-common-subsequence heuristic for finding operational 
        synergy across different execution traces.
        """
        if not traces:
            return []
            
        # Simplified for MVP: Look for exact overlapping sequential pairs/triplets
        # Real implementation would use sequence alignment algorithms (like Smith-Waterman)
        shortest_trace = min(traces, key=len)
        best_subseq = []
        
        for i in range(len(shortest_trace)):
            for j in range(i + 2, len(shortest_trace) + 1): # Minimum 2 steps to be a 'synergy'
                candidate = shortest_trace[i:j]
                
                # Check if candidate exists in all traces
                appears_in_all = True
                for trace in traces:
                    # check sublist
                    found = False
                    for k in range(len(trace) - len(candidate) + 1):
                        if trace[k:k+len(candidate)] == candidate:
                            found = True
                            break
                    if not found:
                        appears_in_all = False
                        break
                        
                if appears_in_all and len(candidate) > len(best_subseq):
                    best_subseq = candidate
                    
        return best_subseq

    def reflect_and_synthesize(self, historical_traces: List[List[str]]) -> bool:
        """
        Analyzes past execution sequences. If a common pattern is found,
        it uses the LLM to semantically name it and injects it into the YAML manifest.
        """
        print(f"[SleepReflector] Analyzing {len(historical_traces)} historical traces for synergy...")
        
        common_sequence = self._find_common_subsequence(historical_traces)
        if not common_sequence:
            print("[SleepReflector] No structural synergy found.")
            return False
            
        print(f"[SleepReflector] Synergy Discovered: {common_sequence}")
        
        if not self.api_key:
            print("[SleepReflector] No GEMINI_API_KEY found. Skipping semantic LLM synthesis.")
            return False
            
        # Use LLM to give it a semantic name and formalize the HTN chunk
        system_prompt = (
            "You are the Metacognitive Sleep Reflector for Kortex Core. "
            "You have detected a recurring sequence of physical actions across unrelated tasks. "
            "Your job is to invent a new abstract 'Meta-Task' name that semantically describes "
            "this sequence, and output it matching the schema."
        )
        
        user_prompt = f"The recurring sequence is: {common_sequence}"
        
        try:
            new_meta_task = self.client.models.generate_content(
                model=self.model_name,
                contents=[system_prompt, user_prompt],
                response_model=SynthesizedMetaTask,
            )
            
            self._inject_into_manifest(new_meta_task)
            return True
        except Exception as e:
            print(f"[SleepReflector] LLM Synthesis failed: {e}")
            return False
            
    def _inject_into_manifest(self, meta_task: SynthesizedMetaTask):
        print(f"[SleepReflector] Injecting new Meta-Task '{meta_task.meta_task_name}' into manifest.")
        
        new_method = {
            "name": f"m_meta_{meta_task.meta_task_name}",
            "target_task": meta_task.meta_task_name,
            "preconditions": meta_task.preconditions,
            "ordered_subtasks": meta_task.ordered_subtasks
        }
        
        if not os.path.exists(self.manifest_path):
            manifest = {"htn_methods": []}
        else:
            with open(self.manifest_path, 'r') as f:
                manifest = yaml.safe_load(f) or {}
                
        if "htn_methods" not in manifest:
            manifest["htn_methods"] = []
            
        manifest["htn_methods"].append(new_method)
        
        with open(self.manifest_path, 'w') as f:
            yaml.dump(manifest, f, sort_keys=False)
