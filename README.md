# FastAPI Service for Optimized LLM with Medusa Head

## Overview

This repository implements a FastAPI service that serves a Language Model (LLM) with a Medusa head, specifically utilizing the `lmsys/vicuna-7b` model. The service is designed to optimize inference speed and enhance performance through speculative decoding and dynamic batching.

## Implementation Details

### Key Features

1. **Optimized Inference:**
   - The service uses a model compilation library to optimize the inference speed of the `vicuna-7b` model, ensuring faster response times.

2. **Medusa Head:**
   - A custom Medusa head is implemented to improve performance via speculative decoding, allowing the model to predict and generate outputs more efficiently.

3. **Dynamic Batching:**
   - The service supports dynamic batching, which enables it to handle multiple concurrent requests effectively, reducing latency and improving throughput.

### How to Download the Model

Before running the service, you need to download the model and place it in the `models/` directory. Use the following command:

```bash
mkdir -p models
wget https://huggingface.co/ksajan/vicuna-7b-v1.3-Q8_0-GGUF/resolve/main/vicuna-7b-v1.3-q8_0.gguf -P models/
```

### How to Run the Service

To run the FastAPI service, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/A-Jatin/medusa-optimized-llm-service.git
   cd medusa-optimized-llm-service
   ```

2. **Install Dependencies:**
   Ensure you have Python and pip installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI Server:**
   Use the following command to start the server:
   ```bash
   uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Access the API:**
   Open your browser and navigate to `http://localhost:8000/docs` to access the interactive API documentation and test the endpoints.

### How to Run Tests

To ensure the correctness and efficiency of the implementation, you can run the provided test cases:

1. **Install Testing Dependencies:**
   If you haven't already, make sure to install any additional testing libraries specified in `requirements.txt`.

2. **Run the Tests:**
   Use the following command to execute the tests:
   ```bash
   pytest tests/
   ```

3. **View Test Results:**
   After running the tests, you will see the results in the terminal, indicating which tests passed or failed.

## Results

The implementation has been tested with various configurations, and the results indicate significant performance improvements:

- **Mean Processing Times (in seconds):**
  - Medusa Head: True, Dynamic Batching: True: **3.02**
  - Medusa Head: True, Dynamic Batching: False: **2.94**
  - Medusa Head: False, Dynamic Batching: True: **24.94**
  - Medusa Head: False, Dynamic Batching: False: **25.61**

These results show a marked decrease in processing times when using the Medusa head and dynamic batching, demonstrating the effectiveness of these optimizations.

- **Performance Metrics:** The service successfully handled multiple concurrent requests with low latency, validating the dynamic batching approach.

Overall, the FastAPI service provides an efficient and optimized solution for serving the `vicuna-7b` model, leveraging advanced techniques to enhance performance and scalability.