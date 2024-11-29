from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from src.model.llm import LLMService
from src.model.batch_manager import DynamicBatchManager

app = FastAPI(title="LLM Service with Medusa Head")

# Create singleton instances
llm_service = LLMService()
batch_manager = DynamicBatchManager(
    max_batch_size=8,  # Increased for better throughput
    max_wait_time=0.1  # Increased to allow more batching
)

class GenerationRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7
    use_medusa: Optional[bool] = True
    use_batching: Optional[bool] = True

class GenerationResponse(BaseModel):
    text: str
    processing_time: float

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest, background_tasks: BackgroundTasks):
    # Update service settings
    llm_service.use_medusa = request.use_medusa
    batch_manager.use_batching = request.use_batching
    return await batch_manager.process_request(request, llm_service)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)