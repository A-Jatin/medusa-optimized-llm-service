import pytest
import asyncio
import statistics
import json
# tests/conftest.py
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import BackgroundTasks

from src.main import GenerationRequest, generate_text

@pytest.mark.asyncio
async def test_processing_time_combinations():
    # Test prompts - using shorter prompts for more consistent timing
    prompts = [
        "Summarize the benefits of exercise.",
        "Explain photosynthesis briefly.",
        "What is climate change?",
        "How do computers work?",
        "Describe the water cycle."
    ]*2  # Duplicate prompts to create more concurrent requests
    
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
        
        
        # Create tasks for concurrent processing
        async def process_prompt(prompt):
            request = GenerationRequest(
                prompt=prompt,
                max_length=100,
                temperature=0.1,
                use_medusa=config["use_medusa"],
                use_batching=config["use_batching"]
            )
            response = await generate_text(request, background_tasks)
            # assert response["text"] is not None
            assert response["text"] != ""
            return response["processing_time"]
        
        # Process all prompts concurrently
        tasks = [process_prompt(prompt) for prompt in prompts]
        processing_times = await asyncio.gather(*tasks)
        
        # Calculate mean processing time for this configuration
        config_name = f"medusa_{config['use_medusa']}_batch_{config['use_batching']}"
        results[config_name] = statistics.mean(processing_times)
        print(f"Mean processing time for {config_name}: {results[config_name]:.3f} seconds")
        
        # Allow time for cleanup between configurations
        await asyncio.sleep(1)
    
    # Print and save results
    print("\nMean Processing Times:")
    for config, mean_time in results.items():
        print(f"{config}: {mean_time:.3f} seconds")

    with open('processing_times.json', 'w') as f:
        json.dump(results, f)
    
    # Basic assertions
    assert len(results) == 4
    assert all(isinstance(time, float) for time in results.values())
    assert all(time > 0 for time in results.values())
