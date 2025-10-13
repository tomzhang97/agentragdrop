
from dataclasses import dataclass

@dataclass
class Site:
    name: str
    bw_mbps: dict
    rtt_ms: dict

DEFAULT_SITES = {
    "vecdb": Site("vecdb", bw_mbps={"llm-gpu": 1000}, rtt_ms={"llm-gpu": 1.0}),
    "llm-gpu": Site("llm-gpu", bw_mbps={"vecdb": 1000}, rtt_ms={"vecdb": 1.0}),
}

def net_cost(bytes_in: int, src: Site, dst: Site) -> float:
    if src.name == dst.name:
        return 0.0
    bw = max(1.0, src.bw_mbps.get(dst.name, 100.0))
    # Transfer time + Round-trip time
    return (bytes_in / (bw * 125000.0)) * 1000.0 + dst.rtt_ms.get(src.name, 1.0)

def placement_for(node_name: str) -> str:
    return "vecdb" if node_name == "retriever" else "llm-gpu"
