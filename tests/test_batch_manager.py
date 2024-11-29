import pytest
from src.model.batch_manager import DynamicBatchManager
from src.main import GenerationRequest
import asyncio

class MockLLMService:
    async def generate(self, prompts, max_length=None, temperature=None):
        await asyncio.sleep(0.1)  # Simulate processing
        return [f"Generated: {p}" for p in prompts], 0.1

@pytest.fixture
def batch_manager():
    return DynamicBatchManager(max_batch_size=2, max_wait_time=0.2)

@pytest.fixture
def mock_llm_service():
    return MockLLMService()

@pytest.mark.asyncio
async def test_single_request(batch_manager, mock_llm_service):
    request = GenerationRequest(prompt="Test prompt")
    response = await batch_manager.process_request(request, mock_llm_service)
    
    assert "Generated: Test prompt" in response["text"]
    assert response["processing_time"] > 0

@pytest.mark.asyncio
async def test_batch_processing(batch_manager, mock_llm_service):
    requests = [
        GenerationRequest(prompt=f"Test prompt {i}")
        for i in range(3)
    ]
    
    tasks = [
        batch_manager.process_request(req, mock_llm_service)
        for req in requests
    ]
    
    responses = await asyncio.gather(*tasks)
    
    assert len(responses) == len(requests)
    for i, response in enumerate(responses):
        assert f"Generated: Test prompt {i}" in response["text"]
        assert response["processing_time"] > 0