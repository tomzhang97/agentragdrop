
import time
from .pruning import ExecutionCache
from .utils import JsonlLogger

class Node:
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

class ExecutionDAG:
    def __init__(self, cache=None, logger: JsonlLogger | None = None):
        self.nodes = []
        self.cache = cache or ExecutionCache()
        self.logger = logger

    def add(self, node):
        self.nodes.append(node)

    def run(self, inputs, pruner=None, budget_tokens=None, budget_time_ms=None):
        ctx, outputs = dict(inputs), {}
        start_time = time.perf_counter()
        used_tokens = 0

        # Determine the execution plan (which nodes to skip)
        order_to_check = [node.name for node in self.nodes]
        nodes_to_skip = set(pruner.decide(order_to_check, ctx)) if pruner else set()

        for node in self.nodes:
            # --- Anytime Budget Checks ---
            if budget_time_ms and (time.perf_counter() - start_time) * 1000 > budget_time_ms:
                outputs.setdefault("composer", {})["answer"] = "[Anytime time budget reached]"
                break

            # --- Pruning Check ---
            if node.name in nodes_to_skip:
                if self.logger: self.logger.log({"event": "pruned", "node": node.name})
                continue

            # --- Cache Check ---
            cached_result = self.cache.get(node.name, ctx)
            if cached_result is not None:
                out = cached_result
                if self.logger: self.logger.log({"event": "cache_hit", "node": node.name})
            else:
                t0_exec = time.perf_counter()
                out = node.fn(**ctx)
                self.cache.put(node.name, ctx, out)
                if self.logger:
                    self.logger.log({
                        "event": "exec", "node": node.name,
                        "latency_ms": (time.perf_counter() - t0_exec) * 1000.0
                    })

            outputs[node.name] = out
            ctx[node.name] = out
            if node.name == "retriever":
                ctx["evidence"] = out.get("evidence", [])

            # --- Token Budget Check (Post-execution) ---
            if isinstance(out, dict):
                used_tokens += int(out.get("tokens_est", 0))
            if budget_tokens and used_tokens > budget_tokens:
                outputs.setdefault("composer", {})["answer"] = "[Anytime token budget reached]"
                break

        if self.logger:
            self.logger.log({"event": "cache_stats", **self.cache.stats()})

        return outputs
