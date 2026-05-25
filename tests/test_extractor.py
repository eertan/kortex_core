import pytest
import os
from unittest.mock import MagicMock
from kortex.extractor.client import GeminiExtractor
from kortex.extractor.models import HTNLaunchPad, ClarificationRequired

def test_extractor_initialization():
    os.environ.pop("GEMINI_API_KEY", None)
    os.environ["GOOGLE_AI_API_KEY"] = "fake-key"
    extractor = GeminiExtractor()
    assert extractor.api_key == "fake-key"
    assert extractor.model_name == "gemini-3.1-pro-preview"

def test_extractor_mock_call():
    extractor = GeminiExtractor(api_key="mock-key")
    
    # Mock the instructor client response
    mock_response = HTNLaunchPad(
        root_task_name="deliver_package",
        task_parameters={"target": "room_204", "item": "medical_kit"}
    )
    
    extractor.client.models.generate_content = MagicMock(return_value=mock_response)
    
    result = extractor.extract_intent(
        prompt="Please take the medical kit to room 204.",
        available_tasks=["deliver_package", "patrol", "charge"]
    )
    
    assert result.root_task_name == "deliver_package"
    assert result.task_parameters["target"] == "room_204"
    assert result.task_parameters["item"] == "medical_kit"

def test_extractor_clarification_mock():
    extractor = GeminiExtractor(api_key="mock-key")
    
    # Mock the instructor client response returning ClarificationRequired
    mock_response = ClarificationRequired(
        question="Are you referring to a household or an individual churn model?",
        reason="Target entity type is ambiguous."
    )
    
    extractor.client.models.generate_content = MagicMock(return_value=mock_response)
    
    result = extractor.extract_intent(
        prompt="Build a churn model.",
        available_tasks=["build_churn_model", "build_ltv_model"]
    )
    
    assert isinstance(result, ClarificationRequired)
    assert "household or an individual" in result.question
    assert "ambiguous" in result.reason
