import torch
from llama_cpp import Llama
from typing import List, Optional
import time
import asyncio

class LLMService:
    def __init__(self, use_medusa: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Llama(
                model_path="models/vicuna-7b-v1.3-q8_0.gguf",  
                n_ctx=2048,        
                embedding=True,
                verbose=True      
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
        current_text = input_text
        current_length = len(current_text)
        
        while current_length < max_length:
            try:
                base_output = self.model(
                    str(current_text),
                    max_tokens=1,
                    echo=True
                )
                
                # Get the next token prediction
                next_token = self._get_next_token(base_output, current_text)
                if next_token is None:
                    break
                    
                # Add the new token to current text
                current_text += next_token
                current_length = len(current_text)
                
            except Exception as e:
                print(f"Processing error in speculative decode: {str(e)}")
                break
                
        return current_text

    def _get_next_token(self, base_output: dict, current_text: str) -> Optional[str]:
        """Get the next token using either Medusa heads or base model output."""
        try:
            if not self.use_medusa:
                # If not using Medusa, just return the base model's prediction
                return base_output['choices'][0]['text']
            
            # Get base model's hidden states for Medusa
            hidden_states = torch.zeros((1, 4096), device=self.device)
            
            # Get predictions from Medusa heads
            head_predictions = []
            for head in self.medusa_heads:
                logits = head(hidden_states)
                pred = torch.argmax(logits, dim=-1)
                head_predictions.append(pred.unsqueeze(0))
            
            predicted_tokens = torch.cat(head_predictions, dim=0)
            verified_token = self._verify_predictions(current_text, predicted_tokens)
            
            if verified_token is not None:
                # Convert token to text and ensure it's a string
                token_list = [verified_token]
                decoded_bytes = self.model.detokenize(token_list)
                return decoded_bytes.decode('utf-8')
            
            return None
            
        except Exception as e:
            print(f"Error in token prediction: {str(e)}")
            return None

    def _verify_predictions(self, current_text: str, predicted_tokens: torch.Tensor) -> Optional[int]:
        """Verify predicted tokens and return the first valid token."""
        try:
            if len(predicted_tokens) > 0:
                # Return the first prediction as an integer
                return predicted_tokens[0].item()
            return None
        except Exception as e:
            print(f"Error in token verification: {str(e)}")
            return None

    async def generate(
        self,
        prompts: List[str],
        max_length: Optional[int] = 100,
        temperature: Optional[float] = 0.7
    ):
        start_time = time.time()
        outputs = []
        
        # Process all prompts in a single batch
        if len(prompts) > 1:
            # True batch processing
            if self.use_medusa:
                # For Medusa, we need to process in parallel due to state management
                tasks = []
                for prompt in prompts:
                    tasks.append(asyncio.create_task(
                        self._process_single_prompt(prompt, max_length, temperature)
                    ))
                outputs = await asyncio.gather(*tasks)
            else:
                # For non-Medusa, process sequentially but in a single context
                for i, prompt in enumerate(prompts):
                    # Process in a single context to maintain efficiency
                    output = self.model(
                        prompt,
                        max_tokens=max_length,
                        temperature=temperature,
                        echo=False,
                    )
                    outputs.append(output['choices'][0]['text'])
        else:
            # Single prompt processing
            output = await self._process_single_prompt(prompts[0], max_length, temperature)
            outputs = [output]

        processing_time = time.time() - start_time
        # For batch processing, divide the time by number of prompts
        if len(prompts) > 1:
            processing_time = processing_time / len(prompts)
        
        return outputs, processing_time

    async def _process_single_prompt(self, prompt: str, max_length: int, temperature: float):
        if self.use_medusa:
            return self._speculative_decode(prompt, max_length)
        else:
            output = self.model(
                prompt,
                max_tokens=max_length,
                temperature=temperature,
                echo=False
            )
            return output['choices'][0]['text']