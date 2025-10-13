
# AgentRAG-Drop (Updated)

Cost-optimized multi-agent RAG with pruning, caching, anytime execution, and explainable logs. This version includes an enhanced evaluation script for Pareto curve generation and ablation studies.

## Install
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# For GPU, install a CUDA-enabled torch build separately first.
```

## 1) Single Query Run
Test a single question with a specific pruner and custom utility weights.
```bash
python experiments/run_pipeline.py   --question "What are the encryption standards?"   --pruner greedy   --utility-weights "0.8,0.1,0.1"   --budget_time_ms 1500
```
A `plan_card.txt` and `decisions.jsonl` log will be saved in `results/`.

## 2) Evaluation Sweeps (for Pareto Curves)
The main evaluation script can sweep across multiple pruners and budgets to generate data for plotting Quality vs. Cost.
```bash
python experiments/eval.py   --dataset data/sample_dataset.jsonl   --corpus data/sample_docs.json   --pruners "none,heuristic,greedy"   --budget-sweep-tokens "0,400,800,1200"   --out "results/pareto_data.csv"
```
This will run multiple experiments (pruners × budgets). The output `results/pareto_data.csv` will contain aggregated results (avg EM, avg tokens, etc.).

## 3) Stress Test
Check throughput and tail latency.
```bash
python experiments/stress.py --n 100 --concurrency 8
```
