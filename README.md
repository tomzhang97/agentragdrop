# 🔍 AgentRAGDrop — HotpotQA Evaluation Framework

This repository implements **AgentRAGDrop**, a distributed multi-round RAG evaluation pipeline.  
It supports **sharded inference**, **multi-GPU parallelism**, and **end-to-end HotpotQA evaluation** (answers + supporting facts + latency/tokens).

---

## 🧭 Overview

AgentRAGDrop performs:
- **Multi-round reasoning** on HotpotQA (distractor setting).
- **Distributed/sharded inference** with per-shard outputs.
- **Unified evaluation** of Answer, Supporting Facts (SP), and **joint** metrics.
- **Performance aggregation** across shards (avg latency, avg tokens, throughput, tokens/sec).

---

## ⚙️ Setup

pip install --index-url https://download.pytorch.org/whl/cu126   torch torchvision torchaudio

pip install -r requirements.txt
```
---

## 🚀 Inference (Multi-GPU, Sharded)

Run sharded prediction with parallel GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/predict_hotpot_sharded.py   --dev_json hotpotQA/hotpot_dev_distractor_v1.json   --out_pred runs/pred_hotpot.json   --shard_size 1000   --gpus 0,1,2,3   --concurrency 4   --evidence_k 8   --sp_k 2
```

This produces per-shard files under `runs/_shards/`:

- `pred_<start>_<end>.json`
- `pred_<start>_<end>_metrics.json` (structured perf; preferred)
- `pred_<start>_<end>.json.log` (plain-text perf summary; fallback)

Example log tail:

```
======================================================================
PREDICTION COMPLETE
======================================================================
Predictions: runs/_shards/pred_3000_4000.json
Metrics: runs/_shards/pred_3000_4000_metrics.json

📊 PERFORMANCE:
  Examples: 1000
  Avg Latency: 1318.95ms
  Avg Tokens: 410.89
  Throughput: 0.76 ex/sec
======================================================================
```

---

## 🔗 Merge Shard Predictions

If your run didn’t already merge shards, do:

```bash
python - <<'PY'
import json, glob
merged={}
for p in sorted(glob.glob('runs/_shards/pred_*.json')):
    merged.update(json.load(open(p)))
json.dump(merged, open('runs/pred_hotpot.json','w'))
print("Merged -> runs/pred_hotpot.json", len(merged))
PY
```

---

## 🧮 Evaluation (Answer + SP + Joint + Performance)

The evaluator accepts **either argument order** and **multiple prediction formats**.

```bash
# Either order works (auto-detected)
python hotpot_evaluate_with_metrics.py runs/pred_hotpot.json hotpotQA/hotpot_dev_distractor_v1.json
# or
python hotpot_evaluate_with_metrics.py hotpotQA/hotpot_dev_distractor_v1.json runs/pred_hotpot.json
```

### What it computes

| Metric | Description |
|---|---|
| `em`, `f1`, `prec`, `recall` | Answer metrics (normalized) |
| `sp_em`, `sp_f1`, `sp_prec`, `sp_recall` | Supporting facts metrics |
| `joint_em`, `joint_f1`, `joint_prec`, `joint_recall` | Combined correctness (answer × SP) |
| `avg_latency_ms`, `avg_tokens_per_example` | Weighted across shards (by examples) |
| `throughput_examples_per_sec` | Global throughput = total examples / total time |
| `tokens_per_sec` | Total tokens / total time |
| `prompt_tokens_total`, `completion_tokens_total`, `total_tokens` | Summed if available in shard metrics |

**Performance aggregation sources (in priority):**
1. Per-shard `pred_*_metrics.json` (structured, preferred)
2. Per-shard `pred_*.json.log` (plain-text summary block)
3. Optional per-prediction JSONL logs placed next to `pred_hotpot.json`

The script aggregates shards **correctly**:
- Example-weighted averages for latency & tokens/example.
- Global throughput: `total_examples / Σ(examples_i / throughput_i)`.
- Tokens/sec: `total_tokens / total_time_s`.

---

## 🧪 Reproduce End-to-End

```bash
# 1) Predict (multi-GPU, sharded)
CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/predict_hotpot_sharded.py   --dev_json data/hotpot_dev_distractor_v1.json   --out_pred runs/pred_hotpot.json   --shard_size 1000   --gpus 0,1,2,3   --concurrency 3   --evidence_k 8 --sp_k 2

# 2) Merge shards (if needed)
python - <<'PY'
import json, glob
merged={}
for p in sorted(glob.glob('runs/_shards/pred_*.json')):
    merged.update(json.load(open(p)))
json.dump(merged, open('runs/pred_hotpot.json','w'))
print("Merged -> runs/pred_hotpot.json", len(merged))
PY

# 3) Evaluate answers + SP + performance
python hotpot_evaluate_with_metrics.py runs/pred_hotpot.json data/hotpot_dev_distractor_v1.json
```

---

## ⚠️ Troubleshooting

- **TypeError: string indices must be integers**  
  You probably passed files in the wrong order or the prediction format is not merged yet. Use the evaluator (it auto-detects), or merge shards as shown above.

- **Missing SP / Joint is low**  
  Ensure predictions include `"sp": [[title, idx], ...]` per id. If not, Answer metrics still work; SP and Joint will drop.

- **Performance fields are zero**  
  Check that `runs/_shards/pred_*_metrics.json` or `pred_*.json.log` exist. The evaluator aggregates both; metrics JSON provides token totals.

---

## 📈 Advanced (Pareto / Ablations)

```bash
python experiments/eval.py   --dataset data/hotpot_dev_distractor_v1.json   --pruners "none,heuristic,greedy"   --budget-sweep-tokens "0,400,800,1200"   --out results/pareto_data.csv
```

---

## 🧠 Notes

- The evaluator accepts prediction shapes:
  - `{"answer": {...}, "sp": {...}}`
  - `{id: "answer"}` or `{id: {"answer": "...", "sp": [...]}}`
  - `[{"id"/"_id": ..., "answer": "...", "sp": [...]}]`
- Cleans common artifacts: leading `"The answer is ..."` and `"U"` → `""`.
- SP items are normalized as `(title, sent_idx)` with zero-based indices.

---

## 📜 License

MIT License © 2025 Tom Zhang.  
Released for academic and research use.
