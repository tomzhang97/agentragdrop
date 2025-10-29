import argparse, os, time, sys
from agentragdrop import (
    ExecutionDAG, Node, RetrieverAgent, ValidatorAgent, CriticAgent, get_llm, utils,
    RAGComposerAgent, HeuristicPruner, RandomPruner, StaticPruner, GreedyPruner,
    EpsilonGreedyPruner
)
from agentragdrop.rag import make_retriever
from agentragdrop.utils import JsonlLogger

# Import central configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

def parse_utility_weights(s: str) -> tuple[float, float, float]:
    try:
        a, b, c = map(float, s.split(','))
        return a, b, c
    except:
        raise argparse.ArgumentTypeError("Utility weights must be three comma-separated floats")

def build_pruner(kind, utility_weights):
    pruner_cls = {
        "heuristic": HeuristicPruner, "random": RandomPruner,
        "static": StaticPruner, "greedy": GreedyPruner, "epsilon": EpsilonGreedyPruner,
    }
    if kind not in pruner_cls:
        return None
    return pruner_cls[kind](utility_weights=utility_weights)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default="Does the contract comply with GDPR?")
    ap.add_argument("--data", default="data/sample_docs.json")
    ap.add_argument("--order", default="rvcc", choices=["rvcc", "rvc", "rc"])
    ap.add_argument("--pruner", default="heuristic", choices=["none", "heuristic", "random", "static", "greedy", "epsilon"])
    ap.add_argument("--utility-weights", type=parse_utility_weights, default="0.6,0.3,0.1")
    ap.add_argument("--k", type=int, default=None)
    # REMOVED --llm_model - use config.py only
    ap.add_argument("--embed_model", default=None)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--budget_tokens", type=int, default=0)
    ap.add_argument("--budget_time_ms", type=int, default=0)
    ap.add_argument("--log_jsonl", default="results/decisions.jsonl")
    ap.add_argument("--out_csv", default="results/metrics.csv")
    ap.add_argument("--plan_card", default="results/plan_card.txt")
    ap.add_argument("--show_config", action="store_true")
    args = ap.parse_args()

    # Use config defaults, NO llm_model override
    llm_model = config.LLM_MODEL  # ALWAYS use config
    embed_model = args.embed_model or config.EMBED_MODEL
    device = args.device if args.device is not None else config.DEFAULT_DEVICE
    k = args.k if args.k is not None else config.RAG_TOP_K

    if args.show_config:
        config.print_config()
        return

    logger = JsonlLogger(args.log_jsonl)
    llm = get_llm(
        model_name=llm_model, 
        device=device,
        max_new_tokens=config.LLM_MAX_NEW_TOKENS,
        temperature=config.LLM_TEMPERATURE,
        do_sample=config.LLM_DO_SAMPLE
    )

    A_r = RetrieverAgent(args.data, embed_model=embed_model, top_k=k)
    A_v = ValidatorAgent(llm)
    A_c = CriticAgent(llm)
    rag_retriever = make_retriever(args.data, embed_model=embed_model, k=k)
    A_p = RAGComposerAgent(rag_retriever, llm)

    dag = ExecutionDAG(logger=logger)
    plan_map = {
        "retriever": lambda question, **_: A_r(question),
        "validator": lambda evidence=None, question=None, **_: A_v(question, evidence),
        "critic": lambda evidence=None, question=None, **_: A_c(question, evidence),
        "composer": lambda question, evidence=None, validator=None, critic=None, **_: A_p(question, evidence)
    }

    plan_order = {
        "rvcc": ["retriever", "validator", "critic", "composer"], 
        "rvc": ["retriever", "validator", "composer"], 
        "rc": ["retriever", "composer"]
    }[args.order]
    
    for name in plan_order:
        dag.add(Node(name, plan_map[name]))

    pruner = build_pruner(args.pruner, args.utility_weights)

    with utils.timer() as t:
        outs = dag.run(
            {"question": args.question}, pruner=pruner,
            budget_tokens=(args.budget_tokens or None),
            budget_time_ms=(args.budget_time_ms or None)
        )
    latency_s = t()

    kept, pruned = [], []
    pruning_rate = 0.0
    if pruner:
        logs = pruner.export_logs()
        kept = [log["node"] for log in logs if log["decision"] == "kept"]
        pruned = [log["node"] for log in logs if log["decision"] == "pruned"]
        total_nodes = len(kept) + len(pruned)
        pruning_rate = (len(pruned) / total_nodes * 100) if total_nodes > 0 else 0

    # Collect metrics
    cache_stats = dag.cache.stats()
    llm_metrics = llm.get_metrics()
    
    # Agent metrics
    agent_metrics = {
        "retriever": A_r.get_metrics() if hasattr(A_r, 'get_metrics') else {},
        "validator": A_v.get_metrics() if hasattr(A_v, 'get_metrics') else {},
        "critic": A_c.get_metrics() if hasattr(A_c, 'get_metrics') else {},
        "composer": A_p.get_metrics() if hasattr(A_p, 'get_metrics') else {},
    }

    ans = outs.get("composer", {}).get("answer", "").strip()

    print("\n" + "="*70)
    print("EXECUTION SUMMARY")
    print("="*70)
    print(f"Model: {llm_model}")
    print(f"Pruner: {args.pruner}, Weights(α,β,γ): {args.utility_weights}")
    print(f"Kept: {kept}")
    print(f"Pruned: {pruned}")
    print(f"Pruning Rate: {pruning_rate:.1f}%")
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"  Total Latency: {latency_s * 1000:.1f}ms")
    
    print(f"\n🔢 TOKEN USAGE:")
    print(f"  Input Tokens: {llm_metrics.get('total_input_tokens', 0)}")
    print(f"  Output Tokens: {llm_metrics.get('total_output_tokens', 0)}")
    print(f"  Total Tokens: {llm_metrics.get('total_tokens', 0)}")
    print(f"  LLM Calls: {llm_metrics.get('total_calls', 0)}")
    print(f"  Avg Tokens/Call: {llm_metrics.get('avg_tokens_per_call', 0):.1f}")
    
    print(f"\n💾 CACHE STATISTICS:")
    print(f"  Cache Queries: {cache_stats.get('cache_queries', 0)}")
    print(f"  Cache Hits: {cache_stats.get('cache_hits', 0)}")
    print(f"  Cache Hit Rate: {cache_stats.get('cache_hit_rate', 0):.1%}")
    
    print(f"\n⏱️  AGENT LATENCIES:")
    for agent_name, metrics in agent_metrics.items():
        if metrics and agent_name in kept:
            print(f"  {agent_name}: {metrics.get('avg_latency_ms', 0):.1f}ms (calls: {metrics.get('total_calls', 0)})")
    
    print(f"\n📝 FINAL ANSWER:")
    print(f"  {ans or '[Empty answer]'}")
    print("="*70)

    utils.write_plan_card(args.plan_card, kept, pruned, outs.get("retriever", {}).get("evidence", []), ans)
    
    # Save detailed metrics
    metrics_output = {
        "config": {
            "model": llm_model,
            "pruner": args.pruner,
            "order": args.order
        },
        "execution": {
            "total_latency_ms": latency_s * 1000,
            "kept_nodes": kept,
            "pruned_nodes": pruned,
            "pruning_rate_pct": pruning_rate
        },
        "tokens": llm_metrics,
        "cache": cache_stats,
        "agents": agent_metrics,
        "answer": ans
    }
    
    metrics_file = args.plan_card.replace(".txt", "_metrics.json")
    import json
    with open(metrics_file, "w") as f:
        json.dump(metrics_output, f, indent=2)
    print(f"\n💾 Detailed metrics saved to: {metrics_file}")

if __name__ == "__main__":
    main()