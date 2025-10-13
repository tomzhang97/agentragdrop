
from .utils import token_estimate
from .rag import make_retriever
from typing import Optional, List, Dict, Any

class AgentResult(dict):
    pass

class RetrieverAgent:
    def __init__(self, data_path, embed_model="sentence-transformers/all-MiniLM-L6-v2", top_k=3):
        self.retriever = make_retriever(data_path, embed_model=embed_model, k=top_k)
        self.vs = self.retriever.vectorstore
        self.top_k = top_k

    def __call__(self, question, k: Optional[int] = None) -> Dict[str, Any]:
        k = k or self.top_k
        results = self.vs.similarity_search_with_score(question, k=k)

        hits, evidence = [], []
        # results: List[(Document, distance)], smaller distance is better
        for doc, dist in results:
            score = 1.0 / (1.0 + float(dist)) # Convert distance to similarity score
            text = doc.page_content
            hits.append(({"text": text}, score))
            evidence.append(text)

        return {"hits": hits, "evidence": evidence, "tokens_est": 0} # Retrieval has no LLM token cost

class ValidatorAgent:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, question, evidence: List[str]):
        prompt = (
            "Answer only YES or NO.\n"
            f"Question: {question}\n"
            "Evidence:\n" + "\n".join(f"- {e[:300]}" for e in (evidence or [])[:3]) + "\n"
            "Is the evidence relevant to the question? Answer YES or NO:"
        )
        out = self.llm.generate(prompt, max_new_tokens=16)
        return AgentResult({"verdict": out.strip(), "tokens_est": token_estimate(prompt + out)})

class CriticAgent:
    def __init__(self, llm, threshold=0.2):
        self.llm = llm
        self.threshold = threshold

    def __call__(self, question, evidence: List[str]):
        notes = ""
        # Rule-based check for very low overlap
        if evidence and len(evidence) >= 2:
            set1, set2 = set(evidence[0].split()), set(evidence[1].split())
            overlap = len(set1.intersection(set2)) / max(1, len(set1.union(set2)))
            if overlap < self.threshold:
                notes = f"Inconsistent evidence (overlap={overlap:.2f})"

        # LLM-based check if no rule triggered
        if not notes:
            prompt = (
                "Critique if the provided evidence items conflict with each other. Be brief.\n"
                f"Q: {question}\nEvidence:\n" + "\n".join(f"- {e[:300]}" for e in (evidence or [])[:3])
            )
            notes = self.llm.generate(prompt, max_new_tokens=64)
            tokens = token_estimate(prompt + notes)
        else:
            tokens = token_estimate(notes)

        return AgentResult({"notes": notes.strip(), "tokens_est": tokens})

class ComposerAgent:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, question, evidence: List[str], validator=None, critic=None):
        prompt = (
            "You are the Composer. Use only the evidence. Return a SHORT PHRASE copied verbatim from the evidence when possible.Do NOT explain.\n"
            f"Question: {question}\n"
            "Evidence:\n" + "\n".join(f"- {e[:400]}" for e in (evidence or [])[:4]) + "\n"
            f"Validator Verdict: {getattr(validator, 'verdict', 'N/A')}\n"
            f"Critic Notes: {getattr(critic, 'notes', 'N/A')}\n"
            "Answer:"
        )
        ans = self.llm.generate(prompt, max_new_tokens=128)
        return AgentResult({"answer": ans.strip(), "tokens_est": token_estimate(prompt + ans)})

class RAGComposerAgent:
    """Composer optimized for HotpotQA - produces concise, direct answers."""
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def __call__(self, question, evidence=None, **kwargs):
        if not evidence:
            docs = self.retriever.get_relevant_documents(question)
            evidence = [d.page_content for d in docs]

        prompt = (
            "Answer the question using ONLY the context below. "
            "Give a SHORT, DIRECT answer (1-5 words). "
            "Do NOT write full sentences or explanations.\n\n"
            f"Question: {question}\n\n"
            "Context:\n" + "\n".join(f"{i+1}. {e[:300]}" for i, e in enumerate((evidence or [])[:5])) + "\n\n"
            "Answer:"
        )
        ans = self.llm.generate(prompt, max_new_tokens=32)  # Reduced from 160
        
        # Clean up the answer - remove citations, extra text
        ans = ans.strip()
        # Remove common prefixes
        for prefix in ["Answer:", "A:", "The answer is", "It is"]:
            if ans.lower().startswith(prefix.lower()):
                ans = ans[len(prefix):].strip()
        
        # Remove citation markers like [1], [2], etc.
        import re
        ans = re.sub(r'\[\d+\]', '', ans).strip()
        
        # Take only first sentence/phrase if model generated too much
        ans = ans.split('.')[0].split('\n')[0].strip()
        
        return {"answer": ans, "tokens_est": token_estimate(prompt + ans)}
