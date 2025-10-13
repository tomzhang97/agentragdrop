
import argparse, concurrent.futures as futures, os, time
import statistics
from agentragdrop import (
    ExecutionDAG, Node, RetrieverAgent, ValidatorAgent, CriticAgent, RAGComposerAgent, get_llm
)
from agentragdrop.rag import make_retriever
from agentragdrop.utils import timer, JsonlLogger

def run_once(dag, pruner, q, budget_tokens=None, budget_time_ms=None):
    with timer() as t:
        dag.run({"question": q}, pruner=pruner, budget_tokens=budget_tokens, budget_time_ms=budget_time_ms)
    return t()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--data", default="data/sample_docs.json")
    ap.add_argument("--llm_model", default="Qwen/Qwen3-1.7B")
    args = ap.parse_args()

    llm = get_llm(model_name=args.llm_model)
    A_r = RetrieverAgent(args.data)
    A_p = RAGComposerAgent(make_retriever(args.data), llm)

    dag = ExecutionDAG()
    dag.add(Node("retriever", A_r))
    dag.add(Node("composer", lambda question, evidence, **_: A_p(question, evidence)))

    queries = [f"Query {i}: What are the key terms of the agreement?" for i in range(args.n)]

    with futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        latencies = list(pool.map(lambda q: run_once(dag, None, q), queries))

    latencies.sort()
    p50 = statistics.median(latencies) * 1000
    p95 = latencies[int(len(latencies) * 0.95)] * 1000
    p99 = latencies[int(len(latencies) * 0.99)] * 1000
    qps = args.n / sum(latencies) if sum(latencies) > 0 else 0

    print(f"Stress test finished ({args.n} queries, concurrency={args.concurrency})")
    print(f"  QPS: {qps:.2f}")
    print(f"  P50 Latency: {p50:.1f} ms")
    print(f"  P95 Latency: {p95:.1f} ms")
    print(f"  P99 Latency: {p99:.1f} ms")

if __name__ == "__main__":
    main()
