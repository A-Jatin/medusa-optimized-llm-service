import pytest
from src.model.llm import LLMService
import torch

@pytest.fixture
def llm_service():
    return LLMService()

def test_medusa_heads_initialization(llm_service):
    assert len(llm_service.medusa_heads) == 4
    for head in llm_service.medusa_heads:
        assert isinstance(head, torch.nn.Linear)

@pytest.mark.asyncio
async def test_text_generation(llm_service):
    prompts = ["Hello, how are"]
    outputs, processing_time = await llm_service.generate(prompts)
    
    assert len(outputs) == len(prompts)
    assert isinstance(outputs[0], str)
    assert len(outputs[0]) > len(prompts[0])
    assert processing_time > 0

@pytest.mark.asyncio
async def test_batch_generation(llm_service):
    prompts = ["Hello,", "Hi there,", "Good morning,"]
    outputs, processing_time = await llm_service.generate(prompts)
    
    assert len(outputs) == len(prompts)
    for output in outputs:
        assert isinstance(output, str)
        assert len(output) > 0