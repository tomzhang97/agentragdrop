# experiments/hotpot_dev_predict_distractor.py
import os, json, argparse, numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from agentragdrop.agents import RAGComposerAgent  # Changed from ComposerAgent
from agentragdrop.llm import get_llm
from agentragdrop.rag import make_retriever  # Add this import
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

def detect_answer_type(question: str) -> str:
    """Detect expected answer type from question."""
    q_lower = question.lower()
    
    # Yes/No questions
    if any(q_lower.startswith(x) for x in ['is ', 'are ', 'was ', 'were ', 'do ', 'does ', 'did ', 'can ', 'could ', 'would ', 'will ']):
        return 'yesno'
    
    # Number/Year questions
    if any(word in q_lower for word in ['how many', 'how much', 'what year', 'when was', 'when did']):
        return 'number'
    
    # Who questions (person names)
    if q_lower.startswith('who '):
        return 'person'
    
    # Where questions (locations)
    if q_lower.startswith('where '):
        return 'location'
    
    return 'entity'  # Default: named entity


def format_answer_by_type(answer: str, answer_type: str) -> str:
    """Post-process answer based on detected type."""
    answer = answer.strip()
    
    if answer_type == 'yesno':
        # Normalize yes/no answers
        if any(word in answer.lower() for word in ['yes', 'correct', 'true', 'affirmative']):
            return 'yes'
        if any(word in answer.lower() for word in ['no', 'not', 'false', 'negative']):
            return 'no'
    
    elif answer_type == 'number':
        # Extract first number
        match = re.search(r'\b\d+\b', answer)
        if match:
            return match.group()
    
    elif answer_type == 'person':
        # Remove titles, keep name proper
        answer = re.sub(r'^(Mr\.|Mrs\.|Ms\.|Dr\.|Professor)\s+', '', answer, flags=re.IGNORECASE)
    
    return answer

def normalize_answer(s: str) -> str:
    s = s.strip()
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def load_dev_distractor(path: str):
    data = json.load(open(path, encoding="utf-8"))
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

def multihop_retrieval(
    q: str,
    sent_triples: List[Tuple[str, str, int]],
    embedder,
    k_per_hop: int = 3,
    num_hops: int = 2
) -> List[Tuple[str, str, int]]:
    """
    Two-stage retrieval for multi-hop questions.
    """
    if not sent_triples:
        return []
    
    sents = [s for _, s, _ in sent_triples]
    s_vec = embedder.encode(sents, normalize_embeddings=True)
    
    # First hop: retrieve based on question
    q_vec = embedder.encode([q], normalize_embeddings=True)
    sims_q = (s_vec @ q_vec.T).reshape(-1)
    top1_idx = np.argsort(-sims_q)[:k_per_hop]
    
    hop1_results = [sent_triples[i] for i in top1_idx]
    hop1_texts = [s for _, s, _ in hop1_results]
    
    # Second hop: retrieve based on first hop + question
    combined_query = q + " " + " ".join(hop1_texts[:2])
    combined_vec = embedder.encode([combined_query], normalize_embeddings=True)
    sims_combined = (s_vec @ combined_vec.T).reshape(-1)
    
    # Get top from second hop, excluding first hop results
    hop1_idx_set = set(top1_idx)
    hop2_candidates = [(i, sims_combined[i]) for i in range(len(sent_triples)) 
                       if i not in hop1_idx_set]
    hop2_candidates.sort(key=lambda x: x[1], reverse=True)
    top2_idx = [i for i, _ in hop2_candidates[:k_per_hop]]
    
    hop2_results = [sent_triples[i] for i in top2_idx]
    
    # Combine and deduplicate by title
    all_results = []
    seen_titles = set()
    
    # Interleave hop1 and hop2 results for diversity
    for i in range(max(len(hop1_results), len(hop2_results))):
        if i < len(hop1_results):
            t, s, idx = hop1_results[i]
            if t not in seen_titles or not t:
                all_results.append(hop1_results[i])
                if t:
                    seen_titles.add(t)
        
        if i < len(hop2_results):
            t, s, idx = hop2_results[i]
            if t not in seen_titles or not t:
                all_results.append(hop2_results[i])
                if t:
                    seen_titles.add(t)
        
        if len(all_results) >= k_per_hop * 2:
            break
    
    return all_results[:k_per_hop * 2]

def build_evidence_texts(top_sent_triples: List[Tuple[str,str,int]]) -> List[str]:
    """Build evidence with better formatting."""
    evidence = []
    for t, s, _ in top_sent_triples:
        s_clean = s.strip()[:250]
        if t:
            evidence.append(f"{t}: {s_clean}")
        else:
            evidence.append(s_clean)
    return evidence

def to_sp(top_sent_triples: List[Tuple[str,str,int]], sp_k=2):
    """Supporting facts in HotpotQA format."""
    return [[t, i] for (t, _, i) in top_sent_triples[:sp_k]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_json", required=True, help="hotpot_dev_distractor_v1.json")
    ap.add_argument("--out_pred", required=True, help="Where to write HotpotQA-format prediction json")
    ap.add_argument("--device", type=int, default=-1, help="-1=cpu, 0=gpu0")
    ap.add_argument("--llm_model", default="meta-llama/Meta-Llama-8B-Instruct")
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--evidence_k", type=int, default=6, help="how many sentences to feed Composer")
    ap.add_argument("--sp_k", type=int, default=2, help="how many sentences to output as supporting facts")
    ap.add_argument("--limit", type=int, default=0, help="limit #examples (0 = all)")
    ap.add_argument("--use_answer_type", action="store_true", help="Use answer type detection")

    args = ap.parse_args()

    print("Loading dev set:", args.dev_json)
    data = load_dev_distractor(args.dev_json)
    if args.limit and args.limit > 0:
        data = data[:args.limit]

    print("Loading embedder:", args.embed_model)
    embedder = SentenceTransformer(args.embed_model, device=("cuda" if args.device != -1 else "cpu"))

    print("Loading LLM:", args.llm_model)
    llm = get_llm(
        model_name=args.llm_model, 
        device=args.device,
        max_new_tokens=32,  # Short answers
        temperature=0.1,    # More deterministic
        do_sample=True
    )
    
    # Create a dummy retriever (won't be used since we pass evidence)
    # We need this because RAGComposerAgent expects a retriever in __init__
    dummy_retriever = None  # We'll pass evidence directly
    composer = RAGComposerAgent(dummy_retriever, llm)

    answer_map, sp_map = {}, {}

    for ex in tqdm(data, total=len(data), desc="Predicting"):
        q = ex["question"]
        _id = ex["_id"]
        sent_triples = flatten_sentences(ex["context"])
        
        # Use multi-hop retrieval
        support = multihop_retrieval(q, sent_triples, embedder, k_per_hop=3, num_hops=2)
        evidence_texts = build_evidence_texts(support)

        # Generate answer
        out = composer(question=q, evidence=evidence_texts)
        raw_ans = (out.get("answer") or "").strip()
        ans = clean_answer(raw_ans)
        
        # Optional: Apply answer type formatting
        if args.use_answer_type:
            answer_type = detect_answer_type(q)
            ans = format_answer_by_type(ans, answer_type)

        # Supporting facts in required format
        sp = to_sp(support, sp_k=args.sp_k)

        answer_map[_id] = ans
        sp_map[_id] = sp

    pred = {"answer": answer_map, "sp": sp_map}
    os.makedirs(os.path.dirname(args.out_pred) or ".", exist_ok=True)
    with open(args.out_pred, "w", encoding="utf-8") as f:
        json.dump(pred, f, ensure_ascii=False)

    print("Wrote predictions to:", args.out_pred)

if __name__ == "__main__":
    main()