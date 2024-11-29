import pytest
import asyncio
import statistics
from src.main import GenerationRequest, generate_text
from fastapi import BackgroundTasks
from src.model.llm import LLMService

@pytest.fixture
def llm_service():
    return LLMService()

@pytest.mark.asyncio
async def test_processing_time_combinations(llm_service):
    # Test prompts
    prompts = [
        "Tell me about artificial intelligence",
        "What is machine learning?",
        "Explain neural networks", 
        "How do transformers work?",
        "Describe deep learning"
    ]
    
    # Configuration combinations
    configs = [
        {"use_medusa": True, "use_batching": True},
        {"use_medusa": True, "use_batching": False},
        {"use_medusa": False, "use_batching": True},
        {"use_medusa": False, "use_batching": False}
    ]
    
    results = {}
    background_tasks = BackgroundTasks()
    
    # Test each configuration
    for config in configs:
        processing_times = []
        
        # Update LLM service settings
        llm_service.use_medusa = config["use_medusa"]
        
        # Process all prompts
        for prompt in prompts:
            request = GenerationRequest(
                prompt=prompt,
                max_length=100,
                temperature=0.1,
                use_medusa=config["use_medusa"],
                use_batching=config["use_batching"]
            )
            
            response = await generate_text(request, background_tasks)
            processing_times.append(response["processing_time"])
        
        # Calculate mean processing time for this configuration
        config_name = f"medusa_{config['use_medusa']}_batch_{config['use_batching']}"
        results[config_name] = statistics.mean(processing_times)
    
    # Print results
    print("\nMean Processing Times:")
    for config, mean_time in results.items():
        print(f"{config}: {mean_time:.3f} seconds")
    
    # Basic assertions
    assert len(results) == 4
    assert all(isinstance(time, float) for time in results.values())
    assert all(time > 0 for time in results.values())
