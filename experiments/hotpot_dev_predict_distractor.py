# experiments/hotpot_dev_predict_distractor.py
import os, json, argparse, numpy as np, time, sys
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from agentragdrop.agents import RAGComposerAgent
from agentragdrop.llm import get_llm
from tqdm import tqdm
import re, string

# Import central configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

def clean_answer(ans: str) -> str:
    """Clean and normalize answer for HotpotQA evaluation."""
    ans = ans.strip()
    
    prefixes = [
        "Answer:", "A:", "The answer is", "It is", "They are",
        "According to the context", "Based on", "The", "Answer is"
    ]
    for prefix in prefixes:
        if ans.lower().startswith(prefix.lower()):
            ans = ans[len(prefix):].strip()
            if ans.startswith(':'):
                ans = ans[1:].strip()
    
    ans = re.sub(r'\[\d+\]', '', ans)
    ans = ans.strip('"\'')
    ans = ans.split('.')[0].split('\n')[0].strip()
    
    while ans and ans[-1] in '.,;:!?':
        ans = ans[:-1].strip()
    
    return ans

def detect_answer_type(question: str) -> str:
    """Detect expected answer type from question."""
    q_lower = question.lower()
    
    if any(q_lower.startswith(x) for x in ['is ', 'are ', 'was ', 'were ', 'do ', 'does ', 'did ', 'can ', 'could ', 'would ', 'will ']):
        return 'yesno'
    
    if any(word in q_lower for word in ['how many', 'how much', 'what year', 'when was', 'when did']):
        return 'number'
    
    if q_lower.startswith('who '):
        return 'person'
    
    if q_lower.startswith('where '):
        return 'location'
    
    return 'entity'

def format_answer_by_type(answer: str, answer_type: str) -> str:
    """Post-process answer based on detected type."""
    answer = answer.strip()
    
    if answer_type == 'yesno':
        if any(word in answer.lower() for word in ['yes', 'correct', 'true', 'affirmative']):
            return 'yes'
        if any(word in answer.lower() for word in ['no', 'not', 'false', 'negative']):
            return 'no'
    
    elif answer_type == 'number':
        match = re.search(r'\b\d+\b', answer)
        if match:
            return match.group()
    
    elif answer_type == 'person':
        answer = re.sub(r'^(Mr\.|Mrs\.|Ms\.|Dr\.|Professor)\s+', '', answer, flags=re.IGNORECASE)
    
    return answer

def load_dev_distractor(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

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
    """Two-stage retrieval for multi-hop questions."""
    if not sent_triples:
        return []
    
    sents = [s for _, s, _ in sent_triples]
    s_vec = embedder.encode(sents, normalize_embeddings=True)
    
    q_vec = embedder.encode([q], normalize_embeddings=True)
    sims_q = (s_vec @ q_vec.T).reshape(-1)
    top1_idx = np.argsort(-sims_q)[:k_per_hop]
    
    hop1_results = [sent_triples[i] for i in top1_idx]
    hop1_texts = [s for _, s, _ in hop1_results]
    
    combined_query = q + " " + " ".join(hop1_texts[:2])
    combined_vec = embedder.encode([combined_query], normalize_embeddings=True)
    sims_combined = (s_vec @ combined_vec.T).reshape(-1)
    
    hop1_idx_set = set(top1_idx)
    hop2_candidates = [(i, sims_combined[i]) for i in range(len(sent_triples)) 
                       if i not in hop1_idx_set]
    hop2_candidates.sort(key=lambda x: x[1], reverse=True)
    top2_idx = [i for i, _ in hop2_candidates[:k_per_hop]]
    
    hop2_results = [sent_triples[i] for i in top2_idx]
    
    all_results = []
    seen_titles = set()
    
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

def select_sp_from_answer(
    answer: str,
    sent_triples: List[Tuple[str, str, int]],
    embedder,
    sp_k: int = 2
) -> List[List]:
    """Select supporting facts that best match the generated answer."""
    if not answer or not sent_triples:
        return [[t, i] for (t, _, i) in sent_triples[:sp_k]]
    
    ans_vec = embedder.encode([answer], normalize_embeddings=True)
    sents = [s for _, s, _ in sent_triples]
    sent_vecs = embedder.encode(sents, normalize_embeddings=True)
    scores = (sent_vecs @ ans_vec.T).reshape(-1)
    
    answer_words = set(answer.lower().split())
    for i, (_, sent, _) in enumerate(sent_triples):
        sent_words = set(sent.lower().split())
        overlap = len(answer_words.intersection(sent_words))
        if overlap > 0:
            scores[i] += 0.3 * (overlap / len(answer_words))
    
    top_idx = np.argsort(-scores)
    sp_facts = []
    used_titles = set()
    
    for idx in top_idx:
        t, s, sent_idx = sent_triples[idx]
        if t not in used_titles or len(sp_facts) >= sp_k - 1:
            sp_facts.append([t, sent_idx])
            if t:
                used_titles.add(t)
        if len(sp_facts) >= sp_k:
            break
    
    return sp_facts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_json", required=True, help="hotpot_dev_distractor_v1.json")
    ap.add_argument("--out_pred", required=True, help="Where to write HotpotQA-format prediction json")
    ap.add_argument("--device", type=int, default=None, help="Device override (-1=cpu, 0+=gpu). Default from config")
    # REMOVED --llm_model - use config.py only
    ap.add_argument("--embed_model", default=None, help="Embed model override. Default from config")
    ap.add_argument("--evidence_k", type=int, default=None, help="Evidence k override. Default from config")
    ap.add_argument("--sp_k", type=int, default=None, help="SP k override. Default from config")
    ap.add_argument("--limit", type=int, default=0, help="limit #examples (0 = all)")
    ap.add_argument("--use_answer_type", action="store_true", default=None, help="Override answer type detection")
    ap.add_argument("--no_answer_type", action="store_true", help="Disable answer type detection")
    ap.add_argument("--use_answer_guided_sp", action="store_true", default=None, help="Override answer-guided SP")
    ap.add_argument("--no_answer_guided_sp", action="store_true", help="Disable answer-guided SP")
    ap.add_argument("--show_config", action="store_true", help="Print configuration and exit")

    args = ap.parse_args()
    
    # Use config defaults, NO llm_model override allowed
    device = args.device if args.device is not None else config.DEFAULT_DEVICE
    llm_model = config.LLM_MODEL  # ALWAYS use config, no override
    embed_model = args.embed_model or config.EMBED_MODEL
    evidence_k = args.evidence_k if args.evidence_k is not None else config.HOTPOT_EVIDENCE_K
    sp_k = args.sp_k if args.sp_k is not None else config.HOTPOT_SP_K
    
    # Handle answer type flags
    if args.no_answer_type:
        use_answer_type = False
    elif args.use_answer_type:
        use_answer_type = True
    else:
        use_answer_type = config.HOTPOT_USE_ANSWER_TYPE
    
    # Handle answer-guided SP flags
    if args.no_answer_guided_sp:
        use_answer_guided_sp = False
    elif args.use_answer_guided_sp:
        use_answer_guided_sp = True
    else:
        use_answer_guided_sp = config.HOTPOT_USE_ANSWER_GUIDED_SP
    
    if args.show_config:
        config.print_config()
        print(f"\nActive settings for this run:")
        print(f"  LLM: {llm_model}")
        print(f"  Embed: {embed_model}")
        print(f"  Evidence K: {evidence_k}, SP K: {sp_k}")
        print(f"  Device: {'GPU ' + str(device) if device >= 0 else 'CPU'}")
        print(f"  Answer Type: {use_answer_type}, Answer-Guided SP: {use_answer_guided_sp}")
        return

    print("="*70)
    print("HOTPOT QA PREDICTION")
    print("="*70)
    print(f"LLM Model: {llm_model}")
    print(f"Embed Model: {embed_model}")
    print(f"Evidence K: {evidence_k}, SP K: {sp_k}")
    print(f"Device: {'GPU ' + str(device) if device >= 0 else 'CPU'}")
    print(f"Answer Type Detection: {use_answer_type}")
    print(f"Answer-Guided SP: {use_answer_guided_sp}")
    print("="*70 + "\n")

    print("Loading dev set:", args.dev_json)
    data = load_dev_distractor(args.dev_json)
    if args.limit and args.limit > 0:
        data = data[:args.limit]

    print("Loading embedder:", embed_model)
    embedder = SentenceTransformer(embed_model, device=("cuda" if device != -1 else "cpu"))

    print("Loading LLM:", llm_model)
    llm = get_llm(
        model_name=llm_model, 
        device=device,
        max_new_tokens=config.LLM_MAX_NEW_TOKENS,
        temperature=config.LLM_TEMPERATURE,
        do_sample=config.LLM_DO_SAMPLE,
        top_p=config.LLM_TOP_P
    )
    
    dummy_retriever = None
    composer = RAGComposerAgent(dummy_retriever, llm)

    answer_map, sp_map = {}, {}
    total_latency = 0.0
    total_tokens = 0
    per_example_metrics = []

    for ex in tqdm(data, total=len(data), desc="Predicting"):
        q = ex["question"]
        _id = ex["_id"]
        
        t_start = time.perf_counter()
        
        sent_triples = flatten_sentences(ex["context"])
        
        t_retrieval_start = time.perf_counter()
        support = multihop_retrieval(q, sent_triples, embedder, 
                                     k_per_hop=config.MULTIHOP_K_PER_HOP, 
                                     num_hops=config.MULTIHOP_NUM_HOPS)
        evidence_texts = build_evidence_texts(support)
        t_retrieval = time.perf_counter() - t_retrieval_start
        
        t_generation_start = time.perf_counter()
        out = composer(question=q, evidence=evidence_texts)
        raw_ans = (out.get("answer") or "").strip()
        ans = clean_answer(raw_ans)
        t_generation = time.perf_counter() - t_generation_start
        
        if use_answer_type:
            answer_type = detect_answer_type(q)
            ans = format_answer_by_type(ans, answer_type)
        
        if use_answer_guided_sp:
            sp = select_sp_from_answer(ans, support, embedder, sp_k=sp_k)
        else:
            sp = [[t, i] for (t, _, i) in support[:sp_k]]
        
        t_total = time.perf_counter() - t_start
        tokens_est = out.get("tokens_est", 0)
        
        total_latency += t_total
        total_tokens += tokens_est
        
        per_example_metrics.append({
            "_id": _id,
            "latency_ms": t_total * 1000,
            "retrieval_ms": t_retrieval * 1000,
            "generation_ms": t_generation * 1000,
            "tokens": tokens_est
        })
        
        answer_map[_id] = ans
        sp_map[_id] = sp

    pred = {"answer": answer_map, "sp": sp_map}
    os.makedirs(os.path.dirname(args.out_pred) or ".", exist_ok=True)
    with open(args.out_pred, "w", encoding="utf-8") as f:
        json.dump(pred, f, ensure_ascii=False)

    metrics_file = args.out_pred.replace(".json", "_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "llm_model": llm_model,
                "embed_model": embed_model,
                "evidence_k": evidence_k,
                "sp_k": sp_k,
                "use_answer_type": use_answer_type,
                "use_answer_guided_sp": use_answer_guided_sp,
                "num_examples": len(data),
                "device": device
            },
            "aggregate": {
                "total_latency_s": round(total_latency, 2),
                "avg_latency_ms": round((total_latency / len(data)) * 1000, 2),
                "total_tokens": total_tokens,
                "avg_tokens_per_example": round(total_tokens / len(data), 2),
                "throughput_examples_per_sec": round(len(data) / total_latency, 2)
            },
            "per_example": per_example_metrics
        }, f, indent=2)

    print("\n" + "="*70)
    print("PREDICTION COMPLETE")
    print("="*70)
    print(f"Predictions: {args.out_pred}")
    print(f"Metrics: {metrics_file}")
    print(f"\n📊 PERFORMANCE:")
    print(f"  Examples: {len(data)}")
    print(f"  Avg Latency: {(total_latency / len(data)) * 1000:.2f}ms")
    print(f"  Avg Tokens: {total_tokens / len(data):.2f}")
    print(f"  Throughput: {len(data) / total_latency:.2f} ex/sec")
    print("="*70)

if __name__ == "__main__":
    main()