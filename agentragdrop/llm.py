
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Optional LangChain wrapper
try:
    from langchain_huggingface import HuggingFacePipeline
except ImportError:
    HuggingFacePipeline = None

class LocalLLM:
    """Thin wrapper over HF causal LM (KISS)."""
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-1.7B",
        device: int = -1,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        do_sample: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if device != -1 and torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        if device != -1 and torch.cuda.is_available():
            self.model = self.model.to(f"cuda:{device}")

        self._pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=(device if device != -1 and torch.cuda.is_available() else -1),
        )

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        out = self._pipe(
            prompt,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )[0]["generated_text"]

        # Return only completion after prompt if possible
        if out.startswith(prompt):
            return out[len(prompt):].strip()
        return out.strip()

def get_llm(model_name: str = "/Qwen/Qwen3-1.7B", device: int = -1, **kw) -> LocalLLM:
    return LocalLLM(model_name=model_name, device=device, **kw)

def get_langchain_llm(model_name: str = "Qwen/Qwen3-1.7B", device: int = -1, **kw):
    if HuggingFacePipeline is None:
        raise ImportError("langchain-huggingface is not installed. Install and retry.")
    llm = LocalLLM(model_name=model_name, device=device, **kw, max_new_tokens=48, temperature=0.1, do_sample=True, top_p=0.9)
    return HuggingFacePipeline(pipeline=llm._pipe)
