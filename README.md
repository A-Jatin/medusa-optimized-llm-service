# LLM FastAPI Service with Medusa Head

⚠️ **Important: Development Environment Requirements**

This project requires a Python environment with GPU support. We recommend using one of these free GPU-enabled environments:

1. **Google Colab**: [Open in Colab](https://colab.research.google.com/)
   - Provides free GPU access
   - Supports all required dependencies
   - Perfect for development and testing

2. **Kaggle Notebooks**: [Open in Kaggle](https://www.kaggle.com/kernels)
   - Offers free GPU runtime
   - Pre-installed ML libraries
   - Great for model training and inference

## Getting Started

1. Choose either Google Colab or Kaggle Notebooks
2. Upload the project files
3. Select GPU runtime in your environment
4. Install dependencies:
   ```python
   !pip install -r requirements.txt
   ```
5. Run the FastAPI service:
   ```python
   !python src/main.py
   ```

## Project Structure

```
├── src/
│   ├── main.py                 # FastAPI service
│   └── model/
│       ├── llm.py             # LLM with Medusa head
│       └── batch_manager.py   # Dynamic batch processing
├── tests/
│   ├── test_llm_service.py
│   └── test_batch_manager.py
└── requirements.txt           # Python dependencies
```

## Features

- FastAPI service for text generation
- Medusa head implementation for speculative decoding
- Dynamic batching for efficient request handling
- Optimized inference using PyTorch
- Comprehensive test suite

## API Endpoints

- POST `/generate`: Generate text from a prompt
- GET `/health`: Health check endpoint

## Implementation Details

### Medusa Head

The implementation uses multiple prediction heads for speculative decoding, which helps improve inference speed by:
- Generating multiple token predictions in parallel
- Verifying predictions with the base model
- Accepting correct predictions to reduce computation

### Dynamic Batching

The batch manager optimizes throughput by:
- Collecting requests into batches
- Processing multiple requests simultaneously
- Balancing batch size and latency

## Performance

The service optimizes performance through:
- Speculative decoding with Medusa heads
- Efficient batch processing
- GPU acceleration when available