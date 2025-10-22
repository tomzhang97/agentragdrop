
import argparse, os, time, itertools
from agentragdrop import (
    ExecutionDAG, Node, RetrieverAgent, ValidatorAgent, CriticAgent, RAGComposerAgent, get_llm, utils,
    HeuristicPruner, RandomPruner, StaticPruner, GreedyPruner, EpsilonGreedyPruner, ExecutionCache
)
from agentragdrop.rag import make_retriever
from agentragdrop.utils import JsonlLogger

def parse_utility_weights(s: str) -> tuple[float, float, float]:
    try:
        a, b, c = map(float, s.split(','))
        return a, b, c
    except:
        raise argparse.ArgumentTypeError("Utility weights must be three comma-separated floats (e.g., '0.6,0.3,0.1')")

def build_pruner(kind, utility_weights):
    pruner_cls = {
        "heuristic": HeuristicPruner, "random": RandomPruner,
        "static": StaticPruner, "greedy": GreedyPruner, "epsilon": EpsilonGreedyPruner
    }
    if kind not in pruner_cls:
        return None
    return pruner_cls[kind](utility_weights=utility_weights)

def run_single_eval(dataset, dag_builder, pruner, budget_tokens, budget_time_ms):
    """Runs one full evaluation for a given configuration."""
    results = []
    dag, cache = dag_builder()
    for ex in dataset:
        q, gold = ex.get("question", ""), ex.get("answer", "")
        if not q: continue

        with utils.timer() as t:
            outs = dag.run(
                {"question": q}, pruner=pruner,
                budget_tokens=budget_tokens, budget_time_ms=budget_time_ms
            )
        latency_s = t()

        pred = outs.get("composer", {}).get("answer", "").strip()
        em = utils.exact_match(pred, gold) if gold else 0
        f1 = utils.f1_score(pred, gold) if gold else 0.0

        pruned_count = 0
        if pruner:
            pruned_count = sum(1 for log in pruner.export_logs() if log['decision'] == 'pruned')
            pruner.reset_logs()

        results.append({
            "question": q, "pred": pred, "latency_s": latency_s,
            "tokens": sum(o.get("tokens_est", 0) for o in outs.values() if isinstance(o, dict)),
            "em": em, "f1": f1, "pruned_count": pruned_count,
            "cache_hits": cache.hits, "cache_queries": cache.queries
        })
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/sample_docs.json")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pruners", default="none,heuristic", help="Comma-separated list of pruners to test")
    ap.add_argument("--order", default="rvcc", choices=["rvcc", "rvc", "rc"])
    ap.add_argument("--utility-weights", type=parse_utility_weights, default="0.6,0.3,0.1")
    ap.add_argument("--budget-sweep-tokens", default="0", help="Comma-separated token budgets (0=unlimited)")
    ap.add_argument("--budget-sweep-time-ms", default="0", help="Comma-separated time budgets in ms (0=unlimited)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--llm_model", default="meta-llama/Meta-Llama-8B-Instruct")
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", type=int, default=-1)
    ap.add_argument("--out", default="results/pareto_data.csv")
    args = ap.parse_args()

    dataset = utils.load_json_or_jsonl(args.dataset)
    pruner_kinds = [p.strip() for p in args.pruners.split(',')]
    token_budgets = [int(b) if int(b) > 0 else None for b in args.budget_sweep_tokens.split(',')]
    time_budgets = [int(b) if int(b) > 0 else None for b in args.budget_sweep_time_ms.split(',')]

    # --- Shared components ---
    llm = get_llm(model_name=args.llm_model, device=args.device)
    A_r = RetrieverAgent(args.corpus, embed_model=args.embed_model, top_k=args.k)
    A_v = ValidatorAgent(llm)
    A_c = CriticAgent(llm)
    rag_retriever = make_retriever(args.corpus, embed_model=args.embed_model, k=args.k)
    A_p = RAGComposerAgent(rag_retriever, llm)

    def dag_builder():
        # Use a fresh cache and logger for each full run to isolate results
        cache = ExecutionCache()
        logger = JsonlLogger(f"results/eval_decisions_{time.time_ns()}.jsonl")
        dag = ExecutionDAG(cache=cache, logger=logger)

        plan_map = {
            "retriever": lambda question, **_: A_r(question),
            "validator": lambda evidence=None, question=None, **_: A_v(question, evidence),
            "critic": lambda evidence=None, question=None, **_: A_c(question, evidence),
            "composer": lambda question, evidence=None, **_: A_p(question, evidence)
        }
        plan_order = {"rvcc": ["retriever", "validator", "critic", "composer"], "rvc": ["retriever", "validator", "composer"], "rc": ["retriever", "composer"]}[args.order]
        for name in plan_order:
            dag.add(Node(name, plan_map[name]))
        return dag, cache

    # --- Main evaluation sweep ---
    all_aggregated_results = []

    for pruner_kind in pruner_kinds:
        for b_tok in token_budgets:
            for b_time in time_budgets:
                print(f"Running eval: pruner='{pruner_kind}', token_budget={b_tok}, time_budget_ms={b_time}...")
                pruner_instance = build_pruner(pruner_kind, args.utility_weights)
                per_example_results = run_single_eval(dataset, dag_builder, pruner_instance, b_tok, b_time)
                summary = utils.aggregate_results(per_example_results)
                a, b, c = args.utility_weights
                summary.update({
                    "pruner": pruner_kind,
                    "token_budget": b_tok or 0,
                    "time_budget_ms": b_time or 0,
                    "utility_weights": f"{a},{b},{c}"
                })
                all_aggregated_results.append(summary)

    # --- Save results ---
    fieldnames = [
        "pruner", "token_budget", "time_budget_ms", "utility_weights",
        "em_avg", "em_stdev", "f1_avg", "f1_stdev", "latency_s_avg", "latency_s_stdev",
        "tokens_avg", "tokens_stdev", "pruned_count_avg",
        "retrieval_recall_avg", "citation_precision_avg", "faithfulness_score_avg"
    ]
    utils.save_csv(args.out, all_aggregated_results, fieldnames)
    print(f"\nSweep finished. Aggregated results saved to {args.out}")

if __name__ == "__main__":
    main()
