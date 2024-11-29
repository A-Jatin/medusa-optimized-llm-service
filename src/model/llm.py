import torch
from llama_cpp import Llama
from typing import List, Optional
import time

class LLMService:
    def __init__(self, use_medusa: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Llama.from_pretrained(
            repo_id="ksajan/vicuna-7b-v1.3-Q8_0-GGUF",
            filename="vicuna-7b-v1.3-q8_0.gguf",
            embedding=True
        )
        self.use_medusa = use_medusa
        self.medusa_heads = self._initialize_medusa_heads() if use_medusa else None

    def _initialize_medusa_heads(self):
        # Initialize Medusa heads for speculative decoding
        num_heads = 4
        heads = []
        for _ in range(num_heads):
            head = torch.nn.Linear(
                4096,  # Vicuna hidden size
                32000  # Vicuna vocab size 
            ).to(self.device)
            heads.append(head)
        return torch.nn.ModuleList(heads)

    def _speculative_decode(self, input_text: str, max_length: int):
        # Implement speculative decoding with Medusa heads
        current_text = input_text
        current_length = len(current_text)
        
        while current_length < max_length:
            # Get base model hidden states
            base_output = self.model.embed(current_text)
            hidden_states = torch.tensor(base_output).to(self.device)
            
            # Reshape hidden states to match expected dimensions
            hidden_states = hidden_states.view(-1, 4096)
            
            # Generate predictions from Medusa heads
            head_predictions = []
            for head in self.medusa_heads:
                logits = head(hidden_states)
                pred = torch.argmax(logits, dim=-1)
                head_predictions.append(pred.unsqueeze(0))
                
            # Verify predictions with base model
            predicted_tokens = torch.cat(head_predictions, dim=0)
            verified_tokens = self._verify_predictions(current_text, predicted_tokens)
            
            # Append verified tokens
            current_text += self.model.detokenize(verified_tokens.cpu().numpy().tolist())
            current_length = len(current_text)
            
        return current_text

    def _verify_predictions(self, input_text: str, predicted_tokens: torch.Tensor):
        # Verify Medusa head predictions with base model
        base_output = self.model(
            input_text,
            max_tokens=100,
            echo=False
        )
        
        # Convert text to tokens using model's tokenizer
        base_text = base_output['choices'][0]['text']
        base_tokens = self.model.tokenize(base_text.encode('utf-8'))  # Ensure string input
        base_token = torch.tensor(base_tokens).to(self.device)
            
        # Compare with Medusa predictions and accept matching ones
        matches = (predicted_tokens == base_token)
        verified_tokens = torch.where(
            matches,
            predicted_tokens,
            base_token
        )
            
        return verified_tokens

    async def generate(
        self,
        prompts: List[str],
        max_length: Optional[int] = 100,
        temperature: Optional[float] = 0.7
    ):
        start_time = time.time()
        
        outputs = []
        for prompt in prompts:
            if self.use_medusa:
                # Generate with speculative decoding
                output_text = self._speculative_decode(
                    prompt,
                    max_length=max_length
                )
            else:
                # Generate without Medusa heads
                output = self.model.generate(
                    prompt,
                    max_tokens=max_length,
                    temperature=temperature,
                    echo=False
                )
                output_text = output['choices'][0]['text']
            outputs.append(output_text)
        
        processing_time = time.time() - start_time
        
        return outputs, processing_time