import os
import asyncio
from typing import Dict, Any, List
from graphiti_core import Graphiti
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.llm_client import LLMClient # Required to initialize if we use Gemini
from kortex.extractor.client import GeminiExtractor

class MemoryManager:
    """
    Manages the dual-phase episodic-semantic memory graph using Graphiti (backed by Kùzu).
    - Phase 1: Fast real-time episodic append (in-memory buffer).
    - Phase 2: Slow asynchronous digestion into Kùzu via Graphiti.
    """

    def __init__(self, db_path: str = "./kortex_kuzu_db"):
        self.db_path = db_path
        self.graphiti = None
        self.episode_buffer: List[Dict[str, Any]] = []

    def _init_client(self):
        if self.graphiti is None:
            # Initialize Graphiti using the local Kùzu Driver
            driver = KuzuDriver(db=self.db_path)
            self.graphiti = Graphiti(graph_driver=driver)

    def hook_post_execution(self, step_name: str, input_payload: dict, execution_result: Any):
        """
        FAST ROUTE HOOK: Triggered immediately after an HTN primitive action executes.
        Appends data onto a flat local memory pipeline buffer.
        """
        episode_log = {
            "source": "execution_spine",
            "step": step_name,
            "context": input_payload,
            "outcome": str(execution_result)
        }
        self.episode_buffer.append(episode_log)

    async def process_sleep_phase(self):
        """
        SLOW ROUTE MAINTENANCE HOOK: Runs asynchronously during background slots.
        Flushes logs to Graphiti/Kuzu, allowing its internal LLM pipeline to build
        temporal relationships, invalidate dead facts, and update schemas.
        """
        if not self.episode_buffer:
            return
            
        self._init_client()
        print(f"[MemoryManager] Digesting {len(self.episode_buffer)} execution episodes into Kùzu Graph...")
        
        while self.episode_buffer:
            episode = self.episode_buffer.pop(0)
            
            content = (f"The agent executed primitive action '{episode['step']}' "
                       f"with parameters: {episode['context']}. "
                       f"The result was: {episode['outcome']}")
            
            # Graphiti parses the string via LLM to extract entities (e.g. 'vault', 'lobby')
            # and builds the temporal knowledge graph inside Kùzu.
            await self.graphiti.add_episode(
                name=f"Execution: {episode['step']}",
                episode_body=content
            )
            
        print("[MemoryManager] Temporal Graphiti/Kùzu consolidation complete.")
