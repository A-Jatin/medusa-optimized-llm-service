from typing import List, Dict, Any
import asyncio
import time
from collections import deque

class DynamicBatchManager:
    def __init__(self, max_batch_size: int = 8, max_wait_time: float = 0.1, use_batching: bool = True):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.request_queue = deque()
        self.processing = False
        self.use_batching = use_batching

    async def process_request(self, request: Any, llm_service: Any):
        if not self.use_batching:
            # Process single request directly
            output, processing_time = await llm_service.generate(
                prompts=[request.prompt],
                max_length=request.max_length,
                temperature=request.temperature
            )
            return {
                "text": output[0],
                "processing_time": processing_time
            }

        # Add request to queue for batched processing
        future = asyncio.Future()
        self.request_queue.append((request, future))
        
        # Start processing if not already running
        if not self.processing:
            self.processing = True
            asyncio.create_task(self._process_batch(llm_service))
        
        # Wait for result
        return await future

    async def _process_batch(self, llm_service: Any):
        while self.request_queue:
            batch = []
            futures = []
            batch_start_time = time.time()
            
            # Collect requests for the batch
            while (
                self.request_queue and 
                len(batch) < self.max_batch_size and 
                (time.time() - batch_start_time) < self.max_wait_time
            ):
                request, future = self.request_queue.popleft()
                batch.append(request)
                futures.append(future)
                
                # Small delay to allow more requests to accumulate
                if len(batch) < self.max_batch_size:
                    await asyncio.sleep(0.001)
            
            # Process the batch
            try:
                prompts = [req.prompt for req in batch]
                # Use the parameters from the first request for consistency
                max_length = batch[0].max_length
                temperature = batch[0].temperature
                
                outputs, processing_time = await llm_service.generate(
                    prompts=prompts,
                    max_length=max_length,
                    temperature=temperature
                )
                
                # Set results for all requests in the batch
                for i, (future, output) in enumerate(zip(futures, outputs)):
                    future.set_result({
                        "text": output,
                        "processing_time": processing_time  # Each request gets the per-item processing time
                    })
                    
            except Exception as e:
                # Handle errors
                for future in futures:
                    if not future.done():
                        future.set_exception(e)
        
        self.processing = False

    async def cleanup(self):
        """Clean up any pending requests and reset processing state."""
        while self.request_queue:
            request, future = self.request_queue.popleft()
            if not future.done():
                future.set_exception(Exception("Batch manager cleanup - request cancelled"))
        self.processing = False