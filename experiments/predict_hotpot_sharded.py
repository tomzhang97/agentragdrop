import os, json, argparse, math, pathlib, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def run_shard(shard_dev, shard_out, gpu_id, evidence_k, sp_k, llm_model):
    """Run one shard on one GPU and log stderr/stdout to a file."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = env.get("PYTHONPATH", os.getcwd())  # <-- ensure package imports

    # Log file per shard
    log_path = shard_out + ".log"
    cmd = [
        sys.executable, "experiments/hotpot_dev_predict_distractor.py",
        "--dev_json", shard_dev,
        "--out_pred", shard_out,
        "--device", "0",                     # inside the child only GPU 0 is visible
        "--evidence_k", str(evidence_k),
        "--sp_k", str(sp_k),
        "--llm_model", llm_model,
    ]
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n")
        lf.flush()
        # Capture output so failures are debuggable
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=lf)
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_json", required=True)
    ap.add_argument("--out_pred", required=True)
    ap.add_argument("--shard_size", type=int, default=1000)
    ap.add_argument("--evidence_k", type=int, default=8)
    ap.add_argument("--sp_k", type=int, default=2)
    ap.add_argument("--llm_model", default="meta-llama/Meta-Llama-8B-Instruct")  # <-- default to Instruct
    ap.add_argument("--gpus", default="0,1,2,3", help="comma-separated GPU ids to use")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    data = load_json(args.dev_json)
    N = len(data)
    shards = math.ceil(N / args.shard_size)

    out_dir = pathlib.Path("runs/_shards")
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_ids:
        raise SystemExit("No GPUs provided via --gpus")

    shard_paths = []
    for s in range(shards):
        lo = s * args.shard_size
        hi = min(N, (s + 1) * args.shard_size)
        shard_dev = out_dir / f"dev_{lo}_{hi}.json"
        shard_out = out_dir / f"pred_{lo}_{hi}.json"
        if not (args.resume and shard_out.exists()):
            with open(shard_dev, "w", encoding="utf-8") as f:
                json.dump(data[lo:hi], f, ensure_ascii=False)
        shard_paths.append((s, lo, hi, str(shard_dev), str(shard_out)))

    # Parallel launch
    futures = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for idx, (s, lo, hi, shard_dev, shard_out) in enumerate(shard_paths):
            if args.resume and os.path.exists(shard_out):
                print(f"[skip] shard {s+1}/{shards} {lo}:{hi} (exists)")
                continue
            gpu = gpu_ids[idx % len(gpu_ids)]
            print(f"[launch] shard {s+1}/{shards} {lo}:{hi} on GPU {gpu}")
            futures.append(pool.submit(
                run_shard, shard_dev, shard_out, gpu, args.evidence_k, args.sp_k, args.llm_model
            ))
        for fut in as_completed(futures):
            fut.result()   # will surface errors; check the .log next to the shard

    # Merge
    merged = {"answer": {}, "sp": {}}
    for _, lo, hi, _, shard_out in shard_paths:
        if not os.path.exists(shard_out):
            raise SystemExit(f"Missing shard output: {shard_out} (see {shard_out}.log)")
        part = load_json(shard_out)
        merged["answer"].update(part.get("answer", {}))
        merged["sp"].update(part.get("sp", {}))

    gold_ids = {ex["_id"] for ex in data}
    miss_a = gold_ids - set(merged["answer"])
    miss_s = gold_ids - set(merged["sp"])
    if miss_a or miss_s:
        some = list((miss_a | miss_s))[:5]
        print("WARNING: missing ids after merge (first few):", some)

    out_path = pathlib.Path(args.out_pred)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)
    print("Wrote merged predictions to:", out_path)

if __name__ == "__main__":
    main()
