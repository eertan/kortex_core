import os
import asyncio
from typing import Dict, Any, List
from graphiti_core import Graphiti
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
        # We need to supply Graphiti with an LLM client to do semantic extraction.
        # It defaults to OpenAI, so for Gemini we'd ideally pass a compatible client,
        # but for this structure we'll configure Graphiti to use the local Kuzu DB.
        
        # Set Kuzu driver URL format
        os.environ["GRAPHITI_NEO4J_URI"] = f"kuzu://{os.path.abspath(self.db_path)}"
        os.environ["GRAPHITI_NEO4J_USER"] = ""
        os.environ["GRAPHITI_NEO4J_PASSWORD"] = ""
        
        self.graphiti = None
        self.episode_buffer: List[Dict[str, Any]] = []

    def _init_client(self):
        if self.graphiti is None:
            # We delay initialization so it doesn't block fast-path execution.
            # Kuzu is a local embedded DB (like SQLite for graphs), so it will
            # create the directory if it doesn't exist.
            self.graphiti = Graphiti(
                neo4j_uri=os.environ["GRAPHITI_NEO4J_URI"],
                neo4j_user="",
                neo4j_password=""
            )

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
