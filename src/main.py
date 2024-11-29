from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from model.llm import LLMService
from model.batch_manager import DynamicBatchManager

app = FastAPI(title="LLM Service with Medusa Head")

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
    # Create service instances with requested settings
    service = LLMService(use_medusa=request.use_medusa)
    manager = DynamicBatchManager(use_batching=request.use_batching)
    return await manager.process_request(request, service)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)