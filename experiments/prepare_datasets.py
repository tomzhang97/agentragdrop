# experiments/prepare_datasets.py
import argparse, json, os, hashlib
from datasets import load_dataset

def ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def jdump(path, obj):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def jwritelines(path, lines):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def _hash(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", "ignore")).hexdigest()[:12]

def _dedup(texts):
    seen, out = set(), []
    for t in texts or []:
        t = (t or "").strip()
        if not t: 
            continue
        h = _hash(t)
        if h in seen:
            continue
        seen.add(h); out.append(t)
    return out

def _mk_corpus(texts):
    texts = _dedup(texts)
    return [{"id": f"doc-{i}", "text": t} for i, t in enumerate(texts)]

def _mk_eval_jsonl(rows):
    return [json.dumps(r, ensure_ascii=False) for r in rows]

# ----------------- LOADERS -----------------

def load_hotpotqa(split="validation", max_ctx_per_page=3, max_ctx_per_example=8):
    """
    HotpotQA 'fullwiki' multi-hop. Handles both pair-format [title, sentences]
    and dict-format {'title': ..., 'sentences': [...]} contexts.
    """
    from datasets import load_dataset
    ds = load_dataset("hotpot_qa", "fullwiki", split=split)

    corpus_texts, eval_rows = [], []

    for ex in ds:
        ctxs = []
        for item in (ex.get("context") or []):
            # Accept both list/tuple pairs and dicts
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title, sents = item[0], item[1]
            elif isinstance(item, dict):
                title = item.get("title", "")
                sents = item.get("sentences") or item.get("context") or item.get("paragraphs") or []
            else:
                continue

            # Normalize sentences to a list of strings
            if not isinstance(sents, list):
                sents = [str(sents)]
            else:
                sents = [str(s) for s in sents if str(s).strip()]

            # Compact per-page chunk for the per-example eval context
            if title or sents:
                compact = f"{title}. " + " ".join(sents[:max_ctx_per_page]) if title else " ".join(sents[:max_ctx_per_page])
                if compact.strip():
                    ctxs.append(compact)

            # Add all sentences to the global corpus
            for s in sents:
                txt = f"{title}. {s}" if title else s
                if txt.strip():
                    corpus_texts.append(txt)

        eval_rows.append({
            "question": ex.get("question", ""),
            "answer": ex.get("answer", ""),
            "context": _dedup(ctxs)[:max_ctx_per_example],
        })

    corpus = _mk_corpus(corpus_texts)
    return corpus, eval_rows


def load_musique(split="validation"):
    ds = load_dataset("yale-nlp/musique", "musique_full_v1.0", split=split)
    corpus_texts, eval_rows = [], []
    for ex in ds:
        q = ex["question"]; ans = ex.get("answer", "")
        ctxs = []
        for p in (ex.get("paragraphs") or []):
            txt = p.get("paragraph_text") or ""
            if txt:
                ctxs.append(txt); corpus_texts.append(txt)
        eval_rows.append({"question": q, "answer": ans, "context": _dedup(ctxs)[:8]})
    return _mk_corpus(corpus_texts), eval_rows

def load_2wiki(split="dev"):
    # Depending on hub mirror, one of these works; try the primary first:
    try:
        ds = load_dataset("DUC2006/2WikiMultihopQA", "2wikimultihopqa", split=split)
    except Exception:
        ds = load_dataset("yale-nlp/2wikimultihopqa", split=split)
    corpus_texts, eval_rows = [], []
    for ex in ds:
        q = ex["question"]; ans = ex.get("answer", "")
        ctxs = []
        for c in ex.get("context", []) or []:
            if isinstance(c, str) and c.strip():
                ctxs.append(c); corpus_texts.append(c)
        eval_rows.append({"question": q, "answer": ans, "context": _dedup(ctxs)[:8]})
    return _mk_corpus(corpus_texts), eval_rows

def load_contractnli(split="validation"):
    ds = load_dataset("nyu-mll/contract_nli", split=split)
    corpus_texts, eval_rows = [], []
    for ex in ds:
        prem = ex.get("premise", "")         # contract text
        hyp  = ex.get("hypothesis", "")      # question (hypothesis)
        lab  = ex.get("label", "Unknown")    # Entailment / Contradiction / Unknown
        ans = {"Entailment":"yes","Contradiction":"no"}.get(lab, "unknown")
        if prem: corpus_texts.append(prem)
        eval_rows.append({"question": hyp, "answer": ans, "context": _dedup([prem])})
    return _mk_corpus(corpus_texts), eval_rows

def load_narrativeqa(split="validation"):
    ds = load_dataset("narrativeqa", split=split)
    corpus_texts, eval_rows = [], []
    for ex in ds:
        # Use summary as the long context; treat Q/A from dataset
        summ = ex.get("summary", "") or ex.get("document", "") or ""
        q = ex.get("question", "")
        ans_list = ex.get("answers") or []
        ans = (ans_list[0].get("text") if ans_list and isinstance(ans_list[0], dict) else "") if ans_list else ""
        if summ: corpus_texts.append(summ)
        eval_rows.append({"question": q, "answer": ans, "context": _dedup([summ])})
    return _mk_corpus(corpus_texts), eval_rows

def load_govreport(split="validation"):
    ds = load_dataset("ccdv/govreport", split=split)
    corpus_texts, eval_rows = [], []
    for ex in ds:
        report = ex.get("report", "") or ""
        summ   = ex.get("summary", "") or ""
        q = "Summarize the key findings of this report."
        if report: corpus_texts.append(report)
        eval_rows.append({"question": q, "answer": summ, "context": _dedup([report])})
    return _mk_corpus(corpus_texts), eval_rows

LOADERS = {
    "hotpotqa":    load_hotpotqa,
    "musique":     load_musique,
    "2wiki":       load_2wiki,
    "contractnli": load_contractnli,
    "narrativeqa": load_narrativeqa,
    "govreport":   load_govreport,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(LOADERS.keys()))
    ap.add_argument("--split", default="validation")  # e.g., validation/dev
    ap.add_argument("--out_prefix", default="data")
    args = ap.parse_args()

    corpus, rows = LOADERS[args.dataset](args.split)

    corpus_path = os.path.join(args.out_prefix, f"{args.dataset}_corpus.json")
    eval_path   = os.path.join(args.out_prefix, f"{args.dataset}_eval.jsonl")

    jdump(corpus_path, corpus)
    jwritelines(eval_path, _mk_eval_jsonl(rows))

    print(f"Wrote:\n  corpus: {corpus_path}\n  eval:   {eval_path}\n  rows:   {len(rows)}  corpus_docs: {len(corpus)}")

if __name__ == "__main__":
    main()
