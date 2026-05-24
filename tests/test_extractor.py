import pytest
import os
from unittest.mock import MagicMock
from kortex.extractor.client import GeminiExtractor
from kortex.extractor.models import HTNLaunchPad

def test_extractor_initialization():
    os.environ["GEMINI_API_KEY"] = "fake-key"
    extractor = GeminiExtractor(model_name="gemini-3.1-pro-preview")
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
