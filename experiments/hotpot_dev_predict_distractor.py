# experiments/hotpot_dev_predict_distractor.py
import os, json, argparse, numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from agentragdrop.agents import ComposerAgent
from agentragdrop.llm import get_llm
from tqdm import tqdm
import re, string

def clean_answer(ans: str) -> str:
    """Clean and normalize answer for HotpotQA evaluation."""
    ans = ans.strip()
    
    # Remove common prefixes
    prefixes = [
        "Answer:", "A:", "The answer is", "It is", "They are",
        "According to the context", "Based on", "The", "Answer is"
    ]
    for prefix in prefixes:
        if ans.lower().startswith(prefix.lower()):
            ans = ans[len(prefix):].strip()
            if ans.startswith(':'):
                ans = ans[1:].strip()
    
    # Remove citation markers [1], [2], etc.
    ans = re.sub(r'\[\d+\]', '', ans)
    
    # Remove quotes if wrapped
    ans = ans.strip('"\'')
    
    # Take first sentence only
    ans = ans.split('.')[0].split('\n')[0].strip()
    
    # Remove trailing punctuation except meaningful ones
    while ans and ans[-1] in '.,;:!?':
        ans = ans[:-1].strip()
    
    return ans

# In your main() function, replace this line:
# ans = (out.get("answer") or "").strip()

# With this:

def normalize_answer(s: str) -> str:
    s = s.strip()
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def diversify_by_title(sent_triples, idx, k):
    # sent_triples[j] = (title, sentence_text, sent_idx)
    seen, out = set(), []
    for j in idx:
        t, s, i = sent_triples[j]
        if t in seen and len(out) < k - 1:
            continue
        out.append((t, s, i))
        seen.add(t)
        if len(out) == k:
            break
    return out if out else [sent_triples[j] for j in idx[:k]]

def window_sentence(page_sents, i, w=1):
    lo, hi = max(0, i-w), min(len(page_sents), i+w+1)
    return " ".join(page_sents[lo:hi])

def load_dev_distractor(path: str):
    data = json.load(open(path, encoding="utf-8"))
    # Each item has: _id, question, context (list of [title, [sentences...]]), answer (string) for dev
    return data

def flatten_sentences(context: List) -> List[Tuple[str, str, int]]:
    """
    context: [[title, [s0, s1, ...]], ...]
    returns list of (title, sentence_text, sentence_idx)
    """
    out = []
    for page in context:
        if isinstance(page, dict):
            title = page.get("title", "")
            sents = page.get("sentences", []) or page.get("context", []) or []
        else:
            title = page[0]
            sents = page[1] if isinstance(page[1], list) else [page[1]]
        for i, s in enumerate(sents):
            s = (s or "").strip()
            if s:
                out.append((title, s, i))
    return out

def topk_supporting_sentences(
    q: str, 
    sent_triples: List[Tuple[str,str,int]], 
    embedder, 
    k=6,  # Increase from 4
    diversity_weight=0.3
) -> List[Tuple[str,str,int]]:
    """
    Retrieve top-k sentences with diversity consideration.
    """
    if not sent_triples:
        return []
    
    sents = [s for _, s, _ in sent_triples]
    q_vec = embedder.encode([q], normalize_embeddings=True)
    s_vec = embedder.encode(sents, normalize_embeddings=True)
    sims = (s_vec @ q_vec.T).reshape(-1)  # cosine similarity
    
    # Get more candidates than needed
    top_n = min(k * 3, len(sent_triples))
    candidate_idx = np.argsort(-sims)[:top_n]
    
    # Diversify by title - ensure we get sentences from different documents
    selected = []
    seen_titles = set()
    
    # First pass: get highest scoring sentence from each unique title
    for idx in candidate_idx:
        title, sent, sent_idx = sent_triples[idx]
        if title not in seen_titles:
            selected.append((idx, sims[idx]))
            seen_titles.add(title)
        if len(selected) >= k:
            break
    
    # Second pass: fill remaining slots with high-scoring sentences
    if len(selected) < k:
        for idx in candidate_idx:
            if idx not in [s[0] for s in selected]:
                selected.append((idx, sims[idx]))
            if len(selected) >= k:
                break
    
    # Return in order of similarity score
    selected.sort(key=lambda x: x[1], reverse=True)
    return [sent_triples[idx] for idx, _ in selected[:k]]


def build_evidence_texts(top_sent_triples: List[Tuple[str,str,int]]) -> List[str]:
    """Build evidence with better formatting."""
    evidence = []
    for t, s, _ in top_sent_triples:
        # Truncate very long sentences but keep important info
        s_clean = s.strip()[:250]  # Reduced from 500
        if t:
            evidence.append(f"{t}: {s_clean}")
        else:
            evidence.append(s_clean)
    return evidence

def to_sp(top_sent_triples: List[Tuple[str,str,int]], sp_k=2):
    # Supporting facts want a *small* set, typically 2 sentences total:
    # [["TitleA", idx], ["TitleB", idx]]
    return [[t, i] for (t, _, i) in top_sent_triples[:sp_k]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_json", required=True, help="hotpot_dev_distractor_v1.json")
    ap.add_argument("--out_pred", required=True, help="Where to write HotpotQA-format prediction json")
    ap.add_argument("--device", type=int, default=-1, help="-1=cpu, 0=gpu0")
    ap.add_argument("--llm_model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--evidence_k", type=int, default=4, help="how many sentences to feed Composer")
    ap.add_argument("--sp_k", type=int, default=2, help="how many sentences to output as supporting facts")
    ap.add_argument("--limit", type=int, default=0, help="limit #examples (0 = all)")

    args = ap.parse_args()

    print("Loading dev set:", args.dev_json)
    data = load_dev_distractor(args.dev_json)
    if args.limit and args.limit > 0:
        data = data[:args.limit]


    print("Loading embedder:", args.embed_model)
    embedder = SentenceTransformer(args.embed_model, device=("cuda" if args.device != -1 else "cpu"))

    print("Loading LLM:", args.llm_model)
    llm = get_llm(model_name=args.llm_model, device=args.device)
    composer = ComposerAgent(llm)

    answer_map, sp_map = {}, {}

    for ex in tqdm(data, total=len(data), desc="Predicting"):
        q = ex["question"]
        _id = ex["_id"]
        sent_triples = flatten_sentences(ex["context"])   # [(title, sent, idx), ...]
        support = topk_supporting_sentences(q, sent_triples, embedder, k=args.evidence_k)
        evidence_texts = build_evidence_texts(support)

        # Compose final answer using your Composer (concise, evidence-bound)
        out = composer(question=q, evidence=evidence_texts)
        raw_ans = (out.get("answer") or "").strip()
        ans = clean_answer(raw_ans)

        # Supporting facts in required format (pick top sp_k)
        sp = to_sp(support, sp_k=args.sp_k)

        answer_map[_id] = ans
        sp_map[_id] = sp

    pred = {"answer": answer_map, "sp": sp_map}
    os.makedirs(os.path.dirname(args.out_pred), exist_ok=True)
    with open(args.out_pred, "w", encoding="utf-8") as f:
        json.dump(pred, f, ensure_ascii=False)

    print("Wrote predictions to:", args.out_pred)

if __name__ == "__main__":
    main()
