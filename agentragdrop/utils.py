
import os, csv, time, json, random, re
from contextlib import contextmanager

@contextmanager
def timer():
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start

def seed(s=42):
    random.seed(s)

def token_estimate(text: str) -> int:
    if not text:
        return 0
    # rough: ~4 chars per token
    return max(1, len(text) // 4)

def save_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

class JsonlLogger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Clear the log file at the start of a run
        if os.path.exists(path):
            os.remove(path)

    def log(self, obj):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_json_or_jsonl(path):
    with open(path, encoding="utf-8") as f:
        first = f.read(1); f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]

def normalize_text(s):
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def exact_match(pred, gold):
    return int(normalize_text(pred) == normalize_text(gold))

def f1_score(pred, gold):
    p_tokens = normalize_text(pred).split()
    g_tokens = normalize_text(gold).split()
    if not p_tokens and not g_tokens: return 1.0
    if not p_tokens or not g_tokens: return 0.0

    common_tokens = 0
    g_counts = {}
    for t in g_tokens:
        g_counts[t] = g_counts.get(t, 0) + 1

    for t in p_tokens:
        if g_counts.get(t, 0) > 0:
            common_tokens += 1
            g_counts[t] -= 1

    if common_tokens == 0: return 0.0

    precision = common_tokens / len(p_tokens)
    recall = common_tokens / len(g_tokens)
    return 2 * precision * recall / (precision + recall)

def write_plan_card(path, kept, pruned, evidence, answer):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== PLAN CARD ===\n")
        f.write(f"Kept: {kept}\n")
        f.write(f"Pruned: {pruned}\n\n")
        f.write("Top Evidence:\n")
        for i, e in enumerate(evidence[:3] if evidence else []):
            f.write(f"[{i+1}] {e[:400]}\n\n")
        f.write("Final Answer:\n")
        f.write(answer if answer else "[Empty]")

def aggregate_results(results_list):
    if not results_list:
        return {}

    keys_to_avg = ["em", "f1", "latency_s", "tokens", "pruned_count", "cache_hits", "cache_queries"]

    summary = {}
    for key in self.keys_to_avg if False else keys_to_avg:  # keep linter calm if unused
        values = [r.get(key, 0) for r in results_list]
        if values:
            import statistics as _statistics
            summary[f"{key}_avg"] = round(_statistics.mean(values), 4)
            summary[f"{key}_stdev"] = round(_statistics.stdev(values) if len(values) > 1 else 0, 4)

    # Add placeholders for future metrics
    summary["retrieval_recall_avg"] = 0.0
    summary["citation_precision_avg"] = 0.0
    summary["faithfulness_score_avg"] = 0.0

    return summary
