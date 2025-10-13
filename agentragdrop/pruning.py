
import hashlib, time, random
from typing import List, Dict, Any, Tuple
from .placement import placement_for, DEFAULT_SITES, net_cost

class ExecutionCache:
    def __init__(self):
        self.store = {}
        self.hits, self.queries = 0, 0

    def key(self, node, inputs):
        # A simple key based on node name and a hash of the inputs dictionary
        # Note: This is sensitive to input dict order and types.
        return hashlib.sha1((node + str(sorted(inputs.items()))).encode()).hexdigest()

    def get(self, node, inputs):
        self.queries += 1
        k = self.key(node, inputs)
        if k in self.store:
            self.hits += 1
            return self.store[k]
        return None

    def put(self, node, inputs, result):
        self.store[self.key(node, inputs)] = result

    def stats(self) -> Dict[str, Any]:
        return {
            "cache_queries": self.queries,
            "cache_hits": self.hits,
            "cache_hit_rate": (self.hits / self.queries) if self.queries else 0.0,
        }

# --- Utility Components ---

def _relevance(ctx):
    hits = ctx.get("retriever", {}).get("hits", [])
    return hits[0][1] if hits else 0.0

def _novelty(ctx):
    evidence = ctx.get("retriever", {}).get("evidence", [])
    if len(evidence) < 2:
        return 1.0
    set1, set2 = set(evidence[0].split()), set(evidence[1].split())
    if not set1 or not set2:
        return 1.0
    overlap = len(set1.intersection(set2)) / max(1, len(set1.union(set2)))
    return 1.0 - overlap

def _consistency(ctx):
    critic_notes = ctx.get("critic", {}).get("notes", "")
    if "inconsistent" in critic_notes.lower() or "conflict" in critic_notes.lower():
        return 0.2

    validator_verdict = ctx.get("validator", {}).get("verdict", "")
    if validator_verdict and validator_verdict.strip().lower().startswith("yes"):
        return 0.8

    return 0.5

def _utility(ctx, alpha=0.6, beta=0.3, gamma=0.1):
    rel = _relevance(ctx)
    nov = _novelty(ctx)
    cons = _consistency(ctx)
    return alpha * rel + beta * nov + gamma * cons, {"rel": rel, "nov": nov, "cons": cons}

def _bytes(ctx):
    evidence = ctx.get("retriever", {}).get("evidence", [])
    return sum(len(x.encode("utf-8")) for x in evidence[:3])

def _cost(node, ctx):
    # Base cost estimates (e.g., in ms or arbitrary units)
    base = {"retriever": 10, "validator": 80, "critic": 100, "composer": 150}.get(node, 50)

    # Network cost component
    site_name = placement_for(node)
    other_site = "vecdb" if site_name != "vecdb" else "llm-gpu"
    net = net_cost(_bytes(ctx), DEFAULT_SITES[site_name], DEFAULT_SITES[other_site])

    k = len(ctx.get("retriever", {}).get("evidence", []))
    return base + 3 * k + net

class BasePruner:
    def __init__(self, must_include=("retriever", "composer"), utility_weights=(0.6, 0.3, 0.1)):
        self.must_include = set(must_include)
        self.alpha, self.beta, self.gamma = utility_weights
        self.logs: List[dict] = []

    def reset_logs(self):
        self.logs.clear()

    def export_logs(self) -> List[dict]:
        return list(self.logs)

    def _log(self, node, util, cost, ratio, decision, dt_ms, policy):
        self.logs.append({
            "policy": policy, "node": node, "utility": round(util, 4),
            "cost": round(cost, 2), "ratio": round(ratio, 5), "decision": decision,
            "decision_overhead_ms": round(dt_ms, 3), "site": placement_for(node)
        })

    def decide(self, order, ctx) -> List[str]:
        raise NotImplementedError

class HeuristicPruner(BasePruner):
    def __init__(self, threshold=0.1, lam=0.0, **kw):
        super().__init__(**kw)
        self.threshold = threshold
        self.lam = lam

    def decide(self, order, ctx):
        skip = []
        for n in order:
            t0 = time.perf_counter()
            util, parts = _utility(ctx, self.alpha, self.beta, self.gamma)
            cost = _cost(n, ctx)
            score = util / (1.0 + self.lam * cost)

            keep = (score >= self.threshold) or (n in self.must_include)
            # Safety guardrail example
            if n == "critic" and parts["rel"] < 0.25:
                keep = False

            dt = (time.perf_counter() - t0) * 1000.0
            self._log(n, util, cost, score, "kept" if keep else "pruned", dt, "heuristic")
            if not keep:
                skip.append(n)
        return skip

class GreedyPruner(BasePruner):
    def __init__(self, threshold=0.05, **kw):
        super().__init__(**kw)
        self.threshold = threshold

    def decide(self, order, ctx):
        pruned = []
        for n in order:
            t0 = time.perf_counter()
            u, _ = _utility(ctx, self.alpha, self.beta, self.gamma)
            c = _cost(n, ctx)
            ratio = u / max(1.0, c)
            keep = (ratio >= self.threshold) or (n in self.must_include)
            dt = (time.perf_counter() - t0) * 1000.0
            self._log(n, u, c, ratio, "kept" if keep else "pruned", dt, "greedy")
            if not keep:
                pruned.append(n)
        return pruned

class EpsilonGreedyPruner(BasePruner):
    def __init__(self, epsilon=0.1, threshold=0.06, **kw):
        super().__init__(**kw)
        self.epsilon = epsilon
        self.threshold = threshold

    def decide(self, order, ctx):
        skip = []
        for n in order:
            t0 = time.perf_counter()
            util, _ = _utility(ctx, self.alpha, self.beta, self.gamma)
            cost = _cost(n, ctx)

            # Exploit: keep if benefit/cost is high enough
            exploit = (util / max(1.0, cost)) >= self.threshold
            # Explore: sometimes keep even if it's not optimal
            explore = (random.random() < self.epsilon)

            keep = exploit or explore or (n in self.must_include)

            dt = (time.perf_counter() - t0) * 1000.0
            ratio = util / max(1.0, cost)
            self._log(n, util, cost, ratio, "kept" if keep else "pruned", dt, "epsilon")
            if not keep:
                skip.append(n)
        return skip

class RandomPruner(BasePruner):
    def __init__(self, p=0.3, **kw):
        super().__init__(**kw)
        self.p = p

    def decide(self, order, ctx):
        skip = []
        for n in order:
            t0 = time.perf_counter()
            keep = (random.random() > self.p) or (n in self.must_include)
            dt = (time.perf_counter() - t0) * 1000.0
            self._log(n, 0, 0, 0, "kept" if keep else "pruned", dt, "random")
            if not keep:
                skip.append(n)
        return skip

class StaticPruner(BasePruner):
    def __init__(self, keep_set=("retriever", "composer"), **kw):
        super().__init__(**kw)
        self.keep_set = set(keep_set) | self.must_include

    def decide(self, order, ctx):
        skip = []
        for n in order:
            t0 = time.perf_counter()
            keep = n in self.keep_set
            dt = (time.perf_counter() - t0) * 1000.0
            self._log(n, 0, 0, 0, "kept" if keep else "pruned", dt, "static")
            if not keep:
                skip.append(n)
        return skip
