
import argparse, os, time
from agentragdrop import (
    ExecutionDAG, Node, RetrieverAgent, ValidatorAgent, CriticAgent, get_llm, utils,
    RAGComposerAgent, HeuristicPruner, RandomPruner, StaticPruner, GreedyPruner,
    EpsilonGreedyPruner
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
    ap.add_argument("--utility-weights", type=parse_utility_weights, default="0.6,0.3,0.1", help="Alpha,beta,gamma for utility (e.g., '0.6,0.3,0.1')")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--llm_model", default="meta-llama/Meta-Llama-8B-Instruct")
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", type=int, default=-1, help="-1=cpu, 0=gpu0")
    ap.add_argument("--budget_tokens", type=int, default=0, help="0 => no token budget")
    ap.add_argument("--budget_time_ms", type=int, default=0, help="0 => no time budget")
    ap.add_argument("--log_jsonl", default="results/decisions.jsonl")
    ap.add_argument("--out_csv", default="results/metrics.csv")
    ap.add_argument("--plan_card", default="results/plan_card.txt")
    args = ap.parse_args()

    logger = JsonlLogger(args.log_jsonl)
    llm = get_llm(model_name=args.llm_model, device=args.device)

    A_r = RetrieverAgent(args.data, embed_model=args.embed_model, top_k=args.k)
    A_v = ValidatorAgent(llm)
    A_c = CriticAgent(llm)
    rag_retriever = make_retriever(args.data, embed_model=args.embed_model, k=args.k)
    A_p = RAGComposerAgent(rag_retriever, llm)

    dag = ExecutionDAG(logger=logger)
    plan_map = {
        "retriever": lambda question, **_: A_r(question),
        "validator": lambda evidence=None, question=None, **_: A_v(question, evidence),
        "critic": lambda evidence=None, question=None, **_: A_c(question, evidence),
        "composer": lambda question, evidence=None, validator=None, critic=None, **_: A_p(question, evidence)
    }

    plan_order = {"rvcc": ["retriever", "validator", "critic", "composer"], "rvc": ["retriever", "validator", "composer"], "rc": ["retriever", "composer"]}[args.order]
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
    if pruner:
        logs = pruner.export_logs()
        kept = [log["node"] for log in logs if log["decision"] == "kept"]
        pruned = [log["node"] for log in logs if log["decision"] == "pruned"]

    print("\n=== PLAN CARD ===")
    print(f"Pruner: {args.pruner}, Weights(α,β,γ): {args.utility_weights}")
    print("Kept:", kept or "[]")
    print("Pruned:", pruned or "[]")

    ans = outs.get("composer", {}).get("answer", "").strip()
    print("\n=== FINAL ANSWER ===\n", ans or "[Empty answer]")
    print(f"\nLatency: {latency_s * 1000:.1f} ms")

    utils.write_plan_card(args.plan_card, kept, pruned, outs.get("retriever", {}).get("evidence", []), ans)

if __name__ == "__main__":
    main()
