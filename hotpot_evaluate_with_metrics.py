# # hotpot_evaluate_with_metrics.py
# import sys
# import json
# import os
# import re
# import string
# from collections import Counter

# # --------------------------
# # Text normalization helpers
# # --------------------------
# def normalize_answer(s: str) -> str:
#     if s is None:
#         return ""
#     def remove_articles(text):
#         return re.sub(r'\b(a|an|the)\b', ' ', text)
#     def white_space_fix(text):
#         return ' '.join(text.split())
#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return ''.join(ch for ch in text if ch not in exclude)
#     def lower(text):
#         return text.lower()
#     return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

# def cleanup_pred_answer(ans):
#     """Light cleanup for common artifacts."""
#     if ans is None:
#         return ""
#     a = str(ans).strip()
#     low = a.lower()
#     if low.startswith("the answer is "):
#         a = a[13:].strip(' "')
#     if a == "U":  # your pipeline sometimes emits "U"
#         a = ""
#     return a

# # --------------------------
# # Core metrics
# # --------------------------
# def f1_score(prediction, ground_truth):
#     normalized_prediction = normalize_answer(prediction)
#     normalized_ground_truth = normalize_answer(ground_truth)

#     ZERO = (0.0, 0.0, 0.0)

#     # Strict handling for yes/no/noanswer disagreements
#     yn = {'yes', 'no', 'noanswer'}
#     if normalized_prediction in yn and normalized_prediction != normalized_ground_truth:
#         return ZERO
#     if normalized_ground_truth in yn and normalized_prediction != normalized_ground_truth:
#         return ZERO

#     ptoks = normalized_prediction.split()
#     gtoks = normalized_ground_truth.split()
#     common = Counter(ptoks) & Counter(gtoks)
#     num_same = sum(common.values())
#     if num_same == 0:
#         return ZERO
#     prec = num_same / float(len(ptoks)) if ptoks else 0.0
#     rec  = num_same / float(len(gtoks)) if gtoks else 0.0
#     f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
#     return f1, prec, rec

# def exact_match_score(prediction, ground_truth):
#     return normalize_answer(prediction) == normalize_answer(ground_truth)

# def update_answer(metrics, prediction, gold):
#     em = exact_match_score(prediction, gold)
#     f1, prec, recall = f1_score(prediction, gold)
#     metrics['em'] += float(em)
#     metrics['f1'] += f1
#     metrics['prec'] += prec
#     metrics['recall'] += recall
#     return em, prec, recall

# def update_sp(metrics, prediction_sp, gold_sp):
#     """prediction_sp and gold_sp are lists of [title, sent_idx]."""
#     # Normalize to tuples for set ops; ignore malformed items gracefully
#     def _norm_pairs(x):
#         out = []
#         for it in x:
#             if isinstance(it, (list, tuple)) and len(it) == 2 and isinstance(it[0], str):
#                 try:
#                     out.append((it[0], int(it[1])))
#                 except Exception:
#                     continue
#         return set(out)

#     cur = _norm_pairs(prediction_sp)
#     gold = _norm_pairs(gold_sp)

#     tp = len(cur & gold)
#     fp = len(cur - gold)
#     fn = len(gold - cur)

#     prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
#     recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
#     f1 = (2 * prec * recall / (prec + recall)) if (prec + recall) > 0 else 0.0
#     em = 1.0 if (fp + fn) == 0 else 0.0

#     metrics['sp_em'] += em
#     metrics['sp_f1'] += f1
#     metrics['sp_prec'] += prec
#     metrics['sp_recall'] += recall
#     return em, prec, recall

# # --------------------------
# # Flexible prediction loader
# # --------------------------
# def load_predictions(pred_path):
#     """
#     Returns a tuple (answers_dict, sp_dict) where:
#       answers_dict: { id: "answer string" }
#       sp_dict:      { id: [ [title, idx], ... ] }
#     Supports:
#       1) {"answer": {...}, "sp": {...}}
#       2) {id: "answer"} or {id: {"answer": "...", "sp": [...]}}
#       3) [{"id"/"_id": ..., "answer": "...", "sp": [...]}]
#     """
#     with open(pred_path, "r") as f:
#         P = json.load(f)

#     # Case 1: split top-level
#     if isinstance(P, dict) and ("answer" in P or "sp" in P):
#         answers = {k: cleanup_pred_answer(v) for k, v in P.get("answer", {}).items()}
#         sps = {k: v for k, v in P.get("sp", {}).items()}
#         return answers, sps

#     # Case 2: id -> value dict
#     if isinstance(P, dict):
#         answers, sps = {}, {}
#         for _id, val in P.items():
#             if isinstance(val, str):
#                 answers[_id] = cleanup_pred_answer(val)
#             elif isinstance(val, dict):
#                 if "answer" in val:
#                     answers[_id] = cleanup_pred_answer(val["answer"])
#                 if "sp" in val and isinstance(val["sp"], list):
#                     sps[_id] = val["sp"]
#         return answers, sps

#     # Case 3: list of records
#     if isinstance(P, list):
#         answers, sps = {}, {}
#         for x in P:
#             if not isinstance(x, dict):
#                 continue
#             _id = x.get("id") or x.get("_id")
#             if not _id:
#                 continue
#             if "answer" in x:
#                 answers[_id] = cleanup_pred_answer(x["answer"])
#             elif "pred" in x:  # sometimes stored as 'pred'
#                 answers[_id] = cleanup_pred_answer(x["pred"])
#             if "sp" in x and isinstance(x["sp"], list):
#                 sps[_id] = x["sp"]
#         return answers, sps

#     raise ValueError("Unsupported prediction format in {}".format(pred_path))

# def load_gold(gold_path):
#     """Load Hotpot dev JSON (list of dicts)."""
#     with open(gold_path, "r") as f:
#         G = json.load(f)
#     if not isinstance(G, list):
#         raise ValueError("Gold file must be a list of examples: {}".format(gold_path))
#     return G

# # --------------------------
# # Evaluation driver
# # --------------------------
# def eval_files(arg1, arg2):
#     """
#     Accepts either order:
#       python hotpot_evaluate_with_metrics.py PRED GOLD
#       python hotpot_evaluate_with_metrics.py GOLD PRED
#     We auto-detect and swap if necessary.
#     """
#     # First try interpret (pred, gold)
#     def _try(pred_path, gold_path):
#         preds = load_predictions(pred_path)
#         gold  = load_gold(gold_path)
#         return preds, gold

#     tried = []
#     try:
#         (answers, sp_map), gold = _try(arg1, arg2)
#         pred_path, gold_path = arg1, arg2
#     except Exception as e1:
#         tried.append(("pred=arg1,gold=arg2", str(e1)))
#         # Try swapped
#         try:
#             (answers, sp_map), gold = _try(arg2, arg1)
#             pred_path, gold_path = arg2, arg1
#         except Exception as e2:
#             tried.append(("pred=arg2,gold=arg1", str(e2)))
#             msg = "Could not determine which file is predictions vs gold.\n"
#             msg += "\n".join([f"Attempt {k} failed: {v}" for k, v in tried])
#             raise RuntimeError(msg)

#     # Metrics accumulators
#     metrics = {
#         'em': 0.0, 'f1': 0.0, 'prec': 0.0, 'recall': 0.0,
#         'sp_em': 0.0, 'sp_f1': 0.0, 'sp_prec': 0.0, 'sp_recall': 0.0,
#         'joint_em': 0.0, 'joint_f1': 0.0, 'joint_prec': 0.0, 'joint_recall': 0.0
#     }

#     missing_answer = 0
#     missing_sp = 0

#     for dp in gold:
#         cur_id = dp.get('_id') or dp.get('id')
#         if cur_id is None:
#             # Skip malformed gold rows
#             continue

#         can_eval_joint = True

#         # Answer
#         if cur_id not in answers:
#             missing_answer += 1
#             can_eval_joint = False
#             em = prec = recall = 0.0
#         else:
#             em, prec, recall = update_answer(metrics, answers[cur_id], dp.get('answer', ""))

#         # Supporting facts
#         gold_sp = dp.get('supporting_facts', [])
#         if cur_id not in sp_map:
#             missing_sp += 1
#             can_eval_joint = False
#             sp_em = sp_prec = sp_recall = 0.0
#         else:
#             sp_em, sp_prec, sp_recall = update_sp(metrics, sp_map[cur_id], gold_sp)

#         # Joint
#         if can_eval_joint:
#             joint_prec = prec * sp_prec
#             joint_recall = recall * sp_recall
#             joint_f1 = (2 * joint_prec * joint_recall / (joint_prec + joint_recall)) if (joint_prec + joint_recall) > 0 else 0.0
#             joint_em = em * sp_em
#             metrics['joint_em'] += joint_em
#             metrics['joint_f1'] += joint_f1
#             metrics['joint_prec'] += joint_prec
#             metrics['joint_recall'] += joint_recall

#     N = len(gold) if len(gold) > 0 else 1
#     for k in list(metrics.keys()):
#         metrics[k] /= N

#     # Merge perf metrics if available
#     metrics_file = pred_path.replace(".json", "_metrics.json")
#     if os.path.exists(metrics_file):
#         try:
#             with open(metrics_file, "r") as f:
#                 perf = json.load(f)
#             agg = perf.get('aggregate', {})
#             metrics.update({
#                 'avg_latency_ms': agg.get('avg_latency_ms', 0),
#                 'avg_tokens': agg.get('avg_tokens_per_example', 0),
#                 'throughput_ex_per_sec': agg.get('throughput_examples_per_sec', 0),
#                 'total_latency_s': agg.get('total_latency_s', 0),
#                 'total_tokens': agg.get('total_tokens', 0),
#             })
#         except Exception:
#             pass

#     # Pretty print summary
#     def pct(x): return f"{100.0 * x:.2f}"
#     print("\n=== HotpotQA Evaluation ===")
#     print(f"Gold:        {gold_path}")
#     print(f"Predictions: {pred_path}")
#     print(f"Examples:    {len(gold)}")
#     print(f"Missing Ans / SP: {missing_answer} / {missing_sp}\n")
#     print(f"Answer    — EM: {pct(metrics['em'])}  F1: {pct(metrics['f1'])}  P: {pct(metrics['prec'])}  R: {pct(metrics['recall'])}")
#     print(f"SupFacts  — EM: {pct(metrics['sp_em'])}  F1: {pct(metrics['sp_f1'])}  P: {pct(metrics['sp_prec'])}  R: {pct(metrics['sp_recall'])}")
#     print(f"Joint     — EM: {pct(metrics['joint_em'])}  F1: {pct(metrics['joint_f1'])}  P: {pct(metrics['joint_prec'])}  R: {pct(metrics['joint_recall'])}")

#     # Also print raw dict for scripts
#     print("\nRaw metrics dict:")
#     print(json.dumps(metrics, ensure_ascii=False, indent=2))

#     return metrics

# if __name__ == '__main__':
#     if len(sys.argv) != 3:
#         print("Usage:\n  python hotpot_evaluate_with_metrics.py PRED GOLD\n  or\n  python hotpot_evaluate_with_metrics.py GOLD PRED")
#         sys.exit(1)
#     eval_files(sys.argv[1], sys.argv[2])


# hotpot_evaluate_with_metrics.py
import sys
import json
import os
import re
import glob
import string
from collections import Counter, defaultdict
from statistics import median

# --------------------------
# Text normalization helpers
# --------------------------
def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

def cleanup_pred_answer(ans):
    if ans is None:
        return ""
    a = str(ans).strip()
    low = a.lower()
    if low.startswith("the answer is "):
        a = a[13:].strip(' "')
    if a == "U":
        a = ""
    return a

# --------------------------
# Core metrics
# --------------------------
def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    ZERO = (0.0, 0.0, 0.0)
    yn = {'yes', 'no', 'noanswer'}
    if normalized_prediction in yn and normalized_prediction != normalized_ground_truth:
        return ZERO
    if normalized_ground_truth in yn and normalized_prediction != normalized_ground_truth:
        return ZERO
    ptoks = normalized_prediction.split()
    gtoks = normalized_ground_truth.split()
    common = Counter(ptoks) & Counter(gtoks)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO
    prec = num_same / float(len(ptoks)) if ptoks else 0.0
    rec  = num_same / float(len(gtoks)) if gtoks else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1, prec, rec

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, prec, recall

def update_sp(metrics, prediction_sp, gold_sp):
    def _norm_pairs(x):
        out = []
        for it in x:
            if isinstance(it, (list, tuple)) and len(it) == 2 and isinstance(it[0], str):
                try:
                    out.append((it[0], int(it[1])))
                except Exception:
                    continue
        return set(out)
    cur = _norm_pairs(prediction_sp)
    gold = _norm_pairs(gold_sp)
    tp = len(cur & gold)
    fp = len(cur - gold)
    fn = len(gold - cur)
    prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * recall / (prec + recall)) if (prec + recall) > 0 else 0.0
    em = 1.0 if (fp + fn) == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall

# --------------------------
# Flexible prediction loader
# --------------------------
def load_predictions(pred_path):
    with open(pred_path, "r") as f:
        P = json.load(f)
    if isinstance(P, dict) and ("answer" in P or "sp" in P):
        answers = {k: cleanup_pred_answer(v) for k, v in P.get("answer", {}).items()}
        sps = {k: v for k, v in P.get("sp", {}).items()}
        return answers, sps
    if isinstance(P, dict):
        answers, sps = {}, {}
        for _id, val in P.items():
            if isinstance(val, str):
                answers[_id] = cleanup_pred_answer(val)
            elif isinstance(val, dict):
                if "answer" in val:
                    answers[_id] = cleanup_pred_answer(val["answer"])
                if "sp" in val and isinstance(val["sp"], list):
                    sps[_id] = val["sp"]
        return answers, sps
    if isinstance(P, list):
        answers, sps = {}, {}
        for x in P:
            if not isinstance(x, dict):
                continue
            _id = x.get("id") or x.get("_id")
            if not _id:
                continue
            if "answer" in x:
                answers[_id] = cleanup_pred_answer(x["answer"])
            elif "pred" in x:
                answers[_id] = cleanup_pred_answer(x["pred"])
            if "sp" in x and isinstance(x["sp"], list):
                sps[_id] = x["sp"]
        return answers, sps
    raise ValueError(f"Unsupported prediction format in {pred_path}")

def load_gold(gold_path):
    with open(gold_path, "r") as f:
        G = json.load(f)
    if not isinstance(G, list):
        raise ValueError(f"Gold file must be a list of examples: {gold_path}")
    return G

# --------------------------
# JSONL log ingestion (tokens & latency)
# --------------------------
ID_KEYS = ["id", "_id", "q_id", "example_id", "uid"]
LAT_MS_KEYS = ["latency_ms", "duration_ms", "elapsed_ms"]
LAT_SEC_KEYS = ["latency", "duration"]
PROMPT_KEYS = ["prompt_tokens", "input_tokens", "tokens_prompt"]
COMP_KEYS = ["completion_tokens", "output_tokens", "tokens_completion"]

def _dig(obj, keys):
    for k in keys:
        if isinstance(obj, dict) and k in obj:
            v = obj[k]
            if isinstance(v, (int, float, str)):
                return v
    return None

def _dig_nested(obj, key_groups):
    if not isinstance(obj, dict): 
        return None
    for wrapper in ["usage", "token_usage", "llm"]:
        if wrapper in obj and isinstance(obj[wrapper], dict):
            v = _dig(obj[wrapper], key_groups)
            if v is not None:
                return v
    return None

def _coerce_ms(x):
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if 0 < v < 60:
        return v * 1000.0  # seconds -> ms
    return v

def parse_jsonl_log(log_path):
    per = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0, "latency_ms": 0.0, "calls": 0})
    if not os.path.exists(log_path):
        return None, {}
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            # id
            rec_id = None
            for k in ID_KEYS:
                if k in rec:
                    rec_id = rec[k]
                    break
            if rec_id is None:
                for container in ["example", "data", "meta"]:
                    if container in rec and isinstance(rec[container], dict):
                        for k in ID_KEYS:
                            if k in rec[container]:
                                rec_id = rec[container][k]
                                break
                    if rec_id:
                        break
            if rec_id is None:
                continue
            # tokens
            p = _dig(rec, PROMPT_KEYS) or _dig_nested(rec, PROMPT_KEYS)
            c = _dig(rec, COMP_KEYS)   or _dig_nested(rec, COMP_KEYS)
            try: p = int(p) if p is not None else 0
            except: p = 0
            try: c = int(c) if c is not None else 0
            except: c = 0
            # latency
            lat = _dig(rec, LAT_MS_KEYS) or _dig_nested(rec, LAT_MS_KEYS) \
                or _dig(rec, LAT_SEC_KEYS) or _dig_nested(rec, LAT_SEC_KEYS)
            lat_ms = _coerce_ms(lat) or 0.0
            per[str(rec_id)]["prompt_tokens"] += p
            per[str(rec_id)]["completion_tokens"] += c
            per[str(rec_id)]["latency_ms"] += float(lat_ms)
            per[str(rec_id)]["calls"] += 1
    lat_list = [v["latency_ms"] for v in per.values() if v["latency_ms"] > 0]
    total_prompt = sum(v["prompt_tokens"] for v in per.values())
    total_comp   = sum(v["completion_tokens"] for v in per.values())
    total_tokens = total_prompt + total_comp
    total_latency_ms = sum(lat_list)
    n = len(per) if per else 1
    agg = {
        "examples_in_log": len(per),
        "prompt_tokens_total": total_prompt,
        "completion_tokens_total": total_comp,
        "total_tokens": total_tokens,
        "avg_tokens_per_example": total_tokens / n if n else 0.0,
        "avg_latency_ms": total_latency_ms / n if n else 0.0,
        "p50_latency_ms": median(lat_list) if lat_list else 0.0,
        "p90_latency_ms": percentile(lat_list, 90) if len(lat_list) >= 2 else (lat_list[0] if lat_list else 0.0),
        "p95_latency_ms": percentile(lat_list, 95) if len(lat_list) >= 2 else (lat_list[0] if lat_list else 0.0),
        "max_latency_ms": max(lat_list) if lat_list else 0.0,
        "throughput_examples_per_sec": (len(per) / (total_latency_ms / 1000.0)) if total_latency_ms > 0 else 0.0,
        "tokens_per_sec": (total_tokens / (total_latency_ms / 1000.0)) if total_latency_ms > 0 else 0.0,
    }
    return agg, per

def percentile(arr, p):
    if not arr:
        return 0.0
    arr = sorted(arr)
    k = (len(arr)-1) * (p/100.0)
    f = int(k)
    c = min(f+1, len(arr)-1)
    if f == c:
        return float(arr[int(k)])
    d0 = arr[f] * (c-k)
    d1 = arr[c] * (k-f)
    return float(d0 + d1)

def find_log_candidate(pred_path):
    cand = [
        pred_path.replace(".json", ".json.log"),
        pred_path.replace(".json", ".jsonl"),
        pred_path.replace(".json", "_metrics.jsonl"),
        os.path.join(os.path.dirname(pred_path), "json.log"),
        os.path.join(os.path.dirname(pred_path), "logs.jsonl"),
    ]
    for p in cand:
        if os.path.exists(p):
            return p
    return None

# --------------------------
# Shard aggregation
# --------------------------
_SUMMARY_RE = re.compile(
    r"Examples:\s*(?P<ex>\d+).*?Avg\s+Latency:\s*(?P<lat>[\d.]+)ms.*?Avg\s+Tokens:\s*(?P<tks>[\d.]+).*?Throughput:\s*(?P<thr>[\d.]+)\s*ex/sec",
    re.IGNORECASE | re.DOTALL
)

def parse_text_summary_log(path):
    """Parse your plain-text summary block."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    m = _SUMMARY_RE.search(txt)
    if not m:
        return None
    ex  = int(m.group("ex"))
    lat = float(m.group("lat"))           # ms
    tks = float(m.group("tks"))           # tokens per example
    thr = float(m.group("thr"))           # ex/sec
    return {"examples": ex, "avg_latency_ms": lat, "avg_tokens": tks, "throughput_ex_per_sec": thr}

def load_shard_perf(shards_dir):
    """
    Aggregate across runs/_shards:
      Prefer pred_*_metrics.json (looks like {"aggregate": {...}})
      Fallback to pred_*.json.log summary block (plain text).
    Returns combined dict or None if nothing found.
    """
    if not os.path.isdir(shards_dir):
        return None

    # Try metrics.json first
    metric_files = sorted(glob.glob(os.path.join(shards_dir, "pred_*_metrics.json")))
    entries = []

    for mf in metric_files:
        try:
            data = json.load(open(mf, "r"))
            agg = data.get("aggregate", {})
            ex  = int(agg.get("examples", agg.get("num_examples", 0)) or 0)
            if ex <= 0:
                continue
            lat = float(agg.get("avg_latency_ms", 0.0))
            tks = float(agg.get("avg_tokens_per_example", agg.get("avg_tokens", 0.0)))
            thr = float(agg.get("throughput_examples_per_sec", 0.0))
            ptt = int(agg.get("prompt_tokens_total", 0))
            ctt = int(agg.get("completion_tokens_total", 0))
            ttt = int(agg.get("total_tokens", ptt + ctt))
            entries.append({
                "examples": ex, "avg_latency_ms": lat, "avg_tokens": tks, "throughput_ex_per_sec": thr,
                "prompt_tokens_total": ptt, "completion_tokens_total": ctt, "total_tokens": ttt
            })
        except Exception:
            continue

    # Fallback to text logs when needed
    if not entries:
        text_logs = sorted(glob.glob(os.path.join(shards_dir, "pred_*.json.log")))
        for lp in text_logs:
            rec = parse_text_summary_log(lp)
            if rec:
                rec.update({"prompt_tokens_total": 0, "completion_tokens_total": 0, "total_tokens": 0})
                entries.append(rec)

    if not entries:
        return None

    # Weighted aggregation by examples
    total_ex = sum(e["examples"] for e in entries)
    if total_ex == 0:
        return None

    # Average latency/tokens: weighted by examples
    avg_lat = sum(e["examples"] * e["avg_latency_ms"] for e in entries) / total_ex
    avg_tokens = sum(e["examples"] * e["avg_tokens"] for e in entries) / total_ex

    # Total tokens if available
    prompt_total = sum(e.get("prompt_tokens_total", 0) for e in entries)
    comp_total   = sum(e.get("completion_tokens_total", 0) for e in entries)
    token_total  = sum(e.get("total_tokens", 0) for e in entries)
    if token_total == 0 and (prompt_total or comp_total):
        token_total = prompt_total + comp_total

    # Global throughput: total_ex / sum(time_i), time_i = ex_i / thr_i
    total_time_s = 0.0
    for e in entries:
        thr = e["throughput_ex_per_sec"]
        if thr and thr > 0:
            total_time_s += (e["examples"] / thr)
        else:
            # If throughput missing, approximate time from latency if possible
            if e["avg_latency_ms"] > 0:
                total_time_s += (e["examples"] * (e["avg_latency_ms"] / 1000.0))
    glob_thr = (total_ex / total_time_s) if total_time_s > 0 else 0.0

    tokens_per_sec = (token_total / total_time_s) if total_time_s > 0 and token_total > 0 else 0.0

    return {
        "shards_count": len(entries),
        "examples_total": total_ex,
        "avg_latency_ms": avg_lat,
        "avg_tokens_per_example": avg_tokens,
        "prompt_tokens_total": prompt_total,
        "completion_tokens_total": comp_total,
        "total_tokens": token_total,
        "throughput_examples_per_sec": glob_thr,
        "tokens_per_sec": tokens_per_sec,
    }

# --------------------------
# Evaluation driver
# --------------------------
def eval_files(arg1, arg2):
    # Try (pred,gold) then swap
    def _try(pred_path, gold_path):
        preds = load_predictions(pred_path)
        gold  = load_gold(gold_path)
        return preds, gold

    tried = []
    try:
        (answers, sp_map), gold = _try(arg1, arg2)
        pred_path, gold_path = arg1, arg2
    except Exception as e1:
        tried.append(("pred=arg1,gold=arg2", str(e1)))
        try:
            (answers, sp_map), gold = _try(arg2, arg1)
            pred_path, gold_path = arg2, arg1
        except Exception as e2:
            tried.append(("pred=arg2,gold=arg1", str(e2)))
            msg = "Could not determine which file is predictions vs gold.\n"
            msg += "\n".join([f"Attempt {k} failed: {v}" for k, v in tried])
            raise RuntimeError(msg)

    # Metrics accumulators
    metrics = {
        'em': 0.0, 'f1': 0.0, 'prec': 0.0, 'recall': 0.0,
        'sp_em': 0.0, 'sp_f1': 0.0, 'sp_prec': 0.0, 'sp_recall': 0.0,
        'joint_em': 0.0, 'joint_f1': 0.0, 'joint_prec': 0.0, 'joint_recall': 0.0
    }

    missing_answer = 0
    missing_sp = 0

    for dp in gold:
        cur_id = dp.get('_id') or dp.get('id')
        if cur_id is None:
            continue
        can_eval_joint = True
        # Answer
        if cur_id not in answers:
            missing_answer += 1
            can_eval_joint = False
            em = prec = recall = 0.0
        else:
            em, prec, recall = update_answer(metrics, answers[cur_id], dp.get('answer', ""))
        # SP
        gold_sp = dp.get('supporting_facts', [])
        if cur_id not in sp_map:
            missing_sp += 1
            can_eval_joint = False
            sp_em = sp_prec = sp_recall = 0.0
        else:
            sp_em, sp_prec, sp_recall = update_sp(metrics, sp_map[cur_id], gold_sp)
        # Joint
        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            joint_f1 = (2 * joint_prec * joint_recall / (joint_prec + joint_recall)) if (joint_prec + joint_recall) > 0 else 0.0
            joint_em = em * sp_em
            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_prec'] += joint_prec
            metrics['joint_recall'] += joint_recall

    N = len(gold) if len(gold) > 0 else 1
    for k in list(metrics.keys()):
        metrics[k] /= N

    # ---- Merge *_metrics.json if present next to predictions
    metrics_file = pred_path.replace(".json", "_metrics.json")
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, "r") as f:
                perf = json.load(f)
            agg = perf.get('aggregate', {})
            metrics.update({
                'avg_latency_ms': agg.get('avg_latency_ms', 0),
                'avg_tokens_per_example': agg.get('avg_tokens_per_example', agg.get('avg_tokens', 0)),
                'throughput_examples_per_sec': agg.get('throughput_examples_per_sec', 0),
                'total_latency_s': agg.get('total_latency_s', 0),
                'total_tokens': agg.get('total_tokens', 0),
                'prompt_tokens_total': agg.get('prompt_tokens_total', 0),
                'completion_tokens_total': agg.get('completion_tokens_total', 0),
            })
        except Exception:
            pass

    # ---- Parse *_jsonl logs if present next to predictions
    log_path = find_log_candidate(pred_path)
    if log_path:
        agg, _ = parse_jsonl_log(log_path)
        if agg:
            # Prefer real log numbers (jsonl) over *_metrics.json if both exist
            metrics.update({
                'avg_latency_ms': agg['avg_latency_ms'],
                'p50_latency_ms': agg['p50_latency_ms'],
                'p90_latency_ms': agg['p90_latency_ms'],
                'p95_latency_ms': agg['p95_latency_ms'],
                'max_latency_ms': agg['max_latency_ms'],
                'avg_tokens_per_example': agg['avg_tokens_per_example'],
                'prompt_tokens_total': agg['prompt_tokens_total'],
                'completion_tokens_total': agg['completion_tokens_total'],
                'total_tokens': agg['total_tokens'],
                'throughput_examples_per_sec': agg['throughput_examples_per_sec'],
                'tokens_per_sec': agg['tokens_per_sec'],
                'examples_accounted_in_log': agg['examples_in_log'],
                'json_log_path': log_path,
            })

    # ---- NEW: aggregate all shard metrics in runs/_shards
    shards_dir = os.path.join(os.path.dirname(pred_path), "_shards")
    shard_agg = load_shard_perf(shards_dir)
    if shard_agg:
        # Shard aggregation represents the *whole run*, so prefer it for global view
        metrics.update({
            'shards_dir': shards_dir,
            'shards_count': shard_agg['shards_count'],
            'examples_total': shard_agg['examples_total'],
            'avg_latency_ms': shard_agg['avg_latency_ms'],
            'avg_tokens_per_example': shard_agg['avg_tokens_per_example'],
            'prompt_tokens_total': shard_agg['prompt_tokens_total'],
            'completion_tokens_total': shard_agg['completion_tokens_total'],
            'total_tokens': shard_agg['total_tokens'],
            'throughput_examples_per_sec': shard_agg['throughput_examples_per_sec'],
            'tokens_per_sec': shard_agg['tokens_per_sec'],
        })

    # Pretty print summary
    def pct(x): return f"{100.0 * x:.2f}"
    print("\n=== HotpotQA Evaluation ===")
    print(f"Gold:        {gold_path}")
    print(f"Predictions: {pred_path}")
    print(f"Examples:    {len(gold)}")
    print(f"Missing Ans / SP: {missing_answer} / {missing_sp}\n")
    print(f"Answer    — EM: {pct(metrics['em'])}  F1: {pct(metrics['f1'])}  P: {pct(metrics['prec'])}  R: {pct(metrics['recall'])}")
    print(f"SupFacts  — EM: {pct(metrics['sp_em'])}  F1: {pct(metrics['sp_f1'])}  P: {pct(metrics['sp_prec'])}  R: {pct(metrics['sp_recall'])}")
    print(f"Joint     — EM: {pct(metrics['joint_em'])}  F1: {pct(metrics['joint_f1'])}  P: {pct(metrics['joint_prec'])}  R: {pct(metrics['joint_recall'])}")

    # Also print raw dict for scripts
    print("\nRaw metrics dict:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage:\n  python hotpot_evaluate_with_metrics.py PRED GOLD\n  or\n  python hotpot_evaluate_with_metrics.py GOLD PRED")
        sys.exit(1)
    eval_files(sys.argv[1], sys.argv[2])
