"""
Client module for the Kortex Extractor.

This module provides the GeminiExtractor class which utilizes google-genai
and instructor to parse natural language prompts into a structured HTNLaunchPad.
"""

import os
from typing import Optional

import instructor
from google import genai

from kortex.extractor.models import HTNLaunchPad, ClarificationRequired, IntentExtraction


class GeminiExtractor:
    """
    Extractor client that uses Google's GenAI (Gemini) models to extract
    structured intents (HTNLaunchPad) from natural language prompts,
    or requests clarification if the prompt is ambiguous.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash"
    ) -> None:
        """
        Initialize the GeminiExtractor.

        Args:
            api_key: Google GenAI API key. If not provided, it will be loaded from
                     the GEMINI_API_KEY environment variable.
            model_name: The name of the Gemini model to use for extraction.
        """
        self.api_key: str = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "API key must be provided either as an argument or via the "
                "GEMINI_API_KEY environment variable."
            )
        
        self.model_name: str = model_name
        
        # Initialize the underlying google-genai client and patch it with instructor
        base_client = genai.Client(api_key=self.api_key)
        self.client = instructor.from_genai(
            client=base_client,
            mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS
        )

    def extract_intent(self, prompt: str, available_tasks: list[str]) -> IntentExtraction:
        """
        Extract the root task and parameters from a natural language prompt.
        If the intent is ambiguous or missing required parameters, returns ClarificationRequired.

        Args:
            prompt: The natural language request from the user or system.
            available_tasks: A list of known tasks in the domain manifest.

        Returns:
            An HTNLaunchPad instance or ClarificationRequired instance.
        """
        system_instruction = (
            "You are an Intent and Parameter Extractor for the Kortex Core system. "
            "Your sole purpose is to analyze the user's natural language request and "
            "determine the appropriate root task name and its required parameters. "
            f"Available HTN tasks you can map to: {available_tasks}. "
            "You must not perform logical reasoning or planning. "
            "CRITICAL HITL RULE: If the user's request is ambiguous, or if it is missing "
            "a mandatory parameter required to confidently execute the task (e.g. they didn't "
            "specify if a churn model is for 'household' or 'individual'), DO NOT GUESS. "
            "Instead, return the ClarificationRequired schema asking the user for the missing info."
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                system_instruction,
                prompt
            ],
            response_model=IntentExtraction,
        )
        
        return response
