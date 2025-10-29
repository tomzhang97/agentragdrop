from .utils import token_estimate
from .rag import make_retriever
from typing import Optional, List, Dict, Any
import re
import time

class AgentResult(dict):
    pass

class RetrieverAgent:
    def __init__(self, data_path, embed_model="sentence-transformers/all-MiniLM-L6-v2", top_k=3):
        self.retriever = make_retriever(data_path, embed_model=embed_model, k=top_k)
        self.vs = self.retriever.vectorstore
        self.top_k = top_k
        
        # Metrics
        self.total_calls = 0
        self.total_latency = 0.0

    def __call__(self, question, k: Optional[int] = None) -> Dict[str, Any]:
        t_start = time.perf_counter()
        
        k = k or self.top_k
        results = self.vs.similarity_search_with_score(question, k=k)

        hits, evidence = [], []
        for doc, dist in results:
            score = 1.0 / (1.0 + float(dist))
            text = doc.page_content
            hits.append(({"text": text}, score))
            evidence.append(text)
        
        t_elapsed = time.perf_counter() - t_start
        self.total_calls += 1
        self.total_latency += t_elapsed

        return {
            "hits": hits, 
            "evidence": evidence, 
            "tokens_est": 0,
            "latency_ms": t_elapsed * 1000
        }
    
    def get_metrics(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_latency_s": round(self.total_latency, 2),
            "avg_latency_ms": round((self.total_latency / self.total_calls) * 1000, 2) if self.total_calls > 0 else 0
        }

class ValidatorAgent:
    def __init__(self, llm):
        self.llm = llm
        self.total_calls = 0
        self.total_latency = 0.0

    def __call__(self, question, evidence: List[str]):
        t_start = time.perf_counter()
        
        prompt = (
            "Answer only YES or NO.\n"
            f"Question: {question}\n"
            "Evidence:\n" + "\n".join(f"- {e[:300]}" for e in (evidence or [])[:3]) + "\n"
            "Is the evidence relevant to the question? Answer YES or NO:"
        )
        out = self.llm.generate(prompt, max_new_tokens=16)
        
        t_elapsed = time.perf_counter() - t_start
        self.total_calls += 1
        self.total_latency += t_elapsed
        
        return AgentResult({
            "verdict": out.strip(), 
            "tokens_est": token_estimate(prompt + out),
            "latency_ms": t_elapsed * 1000
        })
    
    def get_metrics(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_latency_s": round(self.total_latency, 2),
            "avg_latency_ms": round((self.total_latency / self.total_calls) * 1000, 2) if self.total_calls > 0 else 0
        }

class CriticAgent:
    def __init__(self, llm, threshold=0.2):
        self.llm = llm
        self.threshold = threshold
        self.total_calls = 0
        self.total_latency = 0.0

    def __call__(self, question, evidence: List[str]):
        t_start = time.perf_counter()
        
        notes = ""
        if evidence and len(evidence) >= 2:
            set1, set2 = set(evidence[0].split()), set(evidence[1].split())
            overlap = len(set1.intersection(set2)) / max(1, len(set1.union(set2)))
            if overlap < self.threshold:
                notes = f"Inconsistent evidence (overlap={overlap:.2f})"

        if not notes:
            prompt = (
                "Critique if the provided evidence items conflict with each other. Be brief.\n"
                f"Q: {question}\nEvidence:\n" + "\n".join(f"- {e[:300]}" for e in (evidence or [])[:3])
            )
            notes = self.llm.generate(prompt, max_new_tokens=64)
            tokens = token_estimate(prompt + notes)
        else:
            tokens = token_estimate(notes)
        
        t_elapsed = time.perf_counter() - t_start
        self.total_calls += 1
        self.total_latency += t_elapsed

        return AgentResult({
            "notes": notes.strip(), 
            "tokens_est": tokens,
            "latency_ms": t_elapsed * 1000
        })
    
    def get_metrics(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_latency_s": round(self.total_latency, 2),
            "avg_latency_ms": round((self.total_latency / self.total_calls) * 1000, 2) if self.total_calls > 0 else 0
        }

class ComposerAgent:
    def __init__(self, llm):
        self.llm = llm
        self.total_calls = 0
        self.total_latency = 0.0

    def __call__(self, question, evidence: List[str], validator=None, critic=None):
        t_start = time.perf_counter()
        
        prompt = (
            "You are the Composer. Use only the evidence. Return a SHORT PHRASE copied verbatim from the evidence when possible. Do NOT explain.\n"
            f"Question: {question}\n"
            "Evidence:\n" + "\n".join(f"- {e[:400]}" for e in (evidence or [])[:4]) + "\n"
            f"Validator Verdict: {getattr(validator, 'verdict', 'N/A')}\n"
            f"Critic Notes: {getattr(critic, 'notes', 'N/A')}\n"
            "Answer:"
        )
        ans = self.llm.generate(prompt, max_new_tokens=128)
        
        t_elapsed = time.perf_counter() - t_start
        self.total_calls += 1
        self.total_latency += t_elapsed
        
        return AgentResult({
            "answer": ans.strip(), 
            "tokens_est": token_estimate(prompt + ans),
            "latency_ms": t_elapsed * 1000
        })
    
    def get_metrics(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_latency_s": round(self.total_latency, 2),
            "avg_latency_ms": round((self.total_latency / self.total_calls) * 1000, 2) if self.total_calls > 0 else 0
        }

class RAGComposerAgent:
    """Composer with few-shot examples for HotpotQA."""
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.total_calls = 0
        self.total_latency = 0.0
        
        self.few_shot_examples = """Examples of good answers:

Q: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
Context: 1. Shirley Temple held the position of U.S. Ambassador... 2. Kiss and Tell starred Shirley Temple as Corliss Archer...
A: U.S. Ambassador

Q: Were Scott Derrickson and Ed Wood of the same nationality?
Context: 1. Scott Derrickson is an American filmmaker. 2. Ed Wood was an American filmmaker...
A: yes

Q: What year was the creator of Vampire: The Masquerade born?
Context: 1. Mark Rein-Hagen created Vampire: The Masquerade. 2. Mark Rein-Hagen was born in 1964...
A: 1964

"""

    def __call__(self, question, evidence=None, **kwargs):
        t_start = time.perf_counter()
        
        if not evidence:
            if self.retriever:
                docs = self.retriever.get_relevant_documents(question)
                evidence = [d.page_content for d in docs]
            else:
                evidence = []

        prompt = (
            "You are answering questions using provided context. "
            "Give ONLY the direct answer - a name, number, yes/no, or short phrase. "
            "Do NOT write explanations or full sentences.\n\n"
            + self.few_shot_examples +
            "Now answer this question:\n\n"
            f"Q: {question}\n"
            "Context:\n" + "\n".join(f"{i+1}. {e[:250]}" for i, e in enumerate((evidence or [])[:6])) + "\n"
            "A:"
        )
        
        ans = self.llm.generate(prompt, max_new_tokens=32)
        ans = self._clean_answer(ans)
        
        t_elapsed = time.perf_counter() - t_start
        self.total_calls += 1
        self.total_latency += t_elapsed
        
        return AgentResult({
            "answer": ans, 
            "tokens_est": token_estimate(prompt + ans),
            "latency_ms": t_elapsed * 1000
        })
    
    def _clean_answer(self, ans: str) -> str:
        """Aggressive cleaning for HotpotQA format."""
        ans = ans.strip()
        
        ans = re.sub(r'\[\d+\]', '', ans)
        ans = re.sub(r'\*+', '', ans)
        
        prefixes = [
            r'^Answer:\s*', r'^A:\s*', r'^The answer is:?\s*',
            r'^Based on (the )?context,?\s*', r'^According to (the )?context,?\s*',
            r'^It is\s*', r'^They are\s*', r'^The\s+'
        ]
        for prefix in prefixes:
            ans = re.sub(prefix, '', ans, flags=re.IGNORECASE)
        
        ans = re.split(r'[.;,\n]', ans)[0].strip()
        ans = ans.strip('"\'')
        ans = re.sub(r'\s+', ' ', ans)
        ans = ans.rstrip('.,;:!?')
        
        return ans
    
    def get_metrics(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_latency_s": round(self.total_latency, 2),
            "avg_latency_ms": round((self.total_latency / self.total_calls) * 1000, 2) if self.total_calls > 0 else 0
        }