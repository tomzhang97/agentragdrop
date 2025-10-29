from typing import Optional
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Optional LangChain wrapper
try:
    from langchain_huggingface import HuggingFacePipeline
except ImportError:
    HuggingFacePipeline = None

class LocalLLM:
    """Thin wrapper over HF causal LM with metrics tracking."""
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-8B-Instruct",
        device: int = -1,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        do_sample: bool = False,
    ):
        print(f"[CHECKPOINT 1/5] Initializing LocalLLM with model: {model_name}", flush=True)
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        
        # Metrics tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.total_latency = 0.0

        print("[CHECKPOINT 2/5] Loading tokenizer...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("[CHECKPOINT 3/5] Tokenizer loaded.", flush=True)

        print("[CHECKPOINT 4/5] Loading model...", flush=True)
        try:
            # Check CUDA availability
            if device != -1 and torch.cuda.is_available():
                print(f"[CHECKPOINT 4.1/5] Using CUDA device: {device}", flush=True)
                dtype = torch.float16
                device_map = None  # We'll manually move to device
            else:
                print("[CHECKPOINT 4.1/5] Using CPU", flush=True)
                dtype = torch.float32
                device_map = None
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                trust_remote_code=True,
                device_map=device_map
            )
            
            # Manually move model to specified device
            if device != -1 and torch.cuda.is_available():
                self.model = self.model.to(f"cuda:{device}")
                print(f"[CHECKPOINT 4.2/5] Model moved to cuda:{device}", flush=True)
            else:
                print("[CHECKPOINT 4.2/5] Model on CPU", flush=True)
                
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load model. Error: {e}", flush=True)
            raise
            
        print("[CHECKPOINT 5/5] Model loaded. Creating pipeline...", flush=True)

        # Pipeline device handling
        if device != -1 and torch.cuda.is_available():
            pipeline_device = device
        else:
            pipeline_device = -1

        self._pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=pipeline_device,
        )
        print("[CHECKPOINT 6/6] Pipeline created. LLM is ready.", flush=True)

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate text and track metrics."""
        t_start = time.perf_counter()
        
        # Count input tokens
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_token_count = input_ids.shape[1]
        
        out = self._pipe(
            prompt,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )[0]["generated_text"]
        
        t_elapsed = time.perf_counter() - t_start

        # Return only completion after prompt if possible
        if out.startswith(prompt):
            completion = out[len(prompt):].strip()
        else:
            completion = out.strip()
        
        # Count output tokens
        output_token_count = len(self.tokenizer.encode(completion))
        
        # Update metrics
        self.total_input_tokens += input_token_count
        self.total_output_tokens += output_token_count
        self.total_calls += 1
        self.total_latency += t_elapsed
        
        return completion
    
    def get_metrics(self) -> dict:
        """Get current metrics."""
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_latency_s": round(self.total_latency, 2),
            "avg_latency_ms": round((self.total_latency / self.total_calls) * 1000, 2) if self.total_calls > 0 else 0,
            "avg_tokens_per_call": round((self.total_input_tokens + self.total_output_tokens) / self.total_calls, 2) if self.total_calls > 0 else 0
        }
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self.total_latency = 0.0

def get_llm(model_name: str = "meta-llama/Meta-Llama-8B-Instruct", device: int = -1, **kw) -> LocalLLM:
    print(f"[CHECKPOINT 0/5] Calling get_llm for model: {model_name}", flush=True)
    return LocalLLM(model_name=model_name, device=device, **kw)

def get_langchain_llm(model_name: str = "meta-llama/Meta-Llama-8B-Instruct", device: int = -1, **kw):
    if HuggingFacePipeline is None:
        raise ImportError("langchain-huggingface is not installed. Install and retry.")
    llm = LocalLLM(model_name=model_name, device=device, **kw, max_new_tokens=48, temperature=0.1, do_sample=True, top_p=0.9)
    return HuggingFacePipeline(pipeline=llm._pipe)