# experiments/tune_hotpot.py
import subprocess
import json
import os
import sys
from datetime import datetime

# Import central configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

# Configuration from central config
DEV_JSON = config.HOTPOT_DEV_JSON
EVAL_SCRIPT = "hotpot_evaluate_v1.py"
LIMIT = 200
DEVICE = config.DEFAULT_DEVICE
LLM_MODEL = config.LLM_MODEL

# Hyperparameter grid
configs = [
    (6, 2, "baseline"),
    (8, 2, "more_evidence"),
    (6, 3, "more_sp"),
    (8, 3, "both_increased"),
    (10, 2, "max_evidence"),
    (4, 2, "minimal"),
]

def run_prediction(evidence_k, sp_k, run_id):
    """Run prediction with given hyperparameters."""
    pred_file = f"predictions/tune_e{evidence_k}_sp{sp_k}_{run_id}.json"
    os.makedirs("predictions", exist_ok=True)
    
    cmd = [
        "python", "experiments/hotpot_dev_predict_distractor.py",
        "--dev_json", DEV_JSON,
        "--out_pred", pred_file,
        "--evidence_k", str(evidence_k),
        "--sp_k", str(sp_k),
        "--limit", str(LIMIT),
        "--device", str(DEVICE)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return pred_file
    except subprocess.CalledProcessError as e:
        print(f"Error running prediction: {e.stderr}")
        return None

def evaluate_predictions(pred_file):
    """Evaluate predictions and return metrics."""
    cmd = ["python", EVAL_SCRIPT, pred_file, DEV_JSON]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        metrics = eval(result.stdout.strip())
        return metrics
    except subprocess.CalledProcessError as e:
        print(f"Error evaluating: {e.stderr}")
        return None
    except Exception as e:
        print(f"Error parsing results: {e}")
        return None

def main():
    print("="*70)
    print("HOTPOT QA HYPERPARAMETER TUNING")
    print("="*70)
    print(f"Model: {LLM_MODEL}")
    print(f"Dataset: {DEV_JSON}")
    print(f"Examples: {LIMIT}")
    print(f"Device: {'GPU ' + str(DEVICE) if DEVICE >= 0 else 'CPU'}")
    print("="*70 + "\n")
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    best_f1 = 0
    best_config = None
    best_metrics = None
    
    for i, (evidence_k, sp_k, desc) in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Testing: {desc}")
        print(f"  evidence_k={evidence_k}, sp_k={sp_k}")
        print("-" * 70)
        
        pred_file = run_prediction(evidence_k, sp_k, run_id)
        if not pred_file:
            print(f"  ❌ Prediction failed, skipping...")
            continue
        
        metrics = evaluate_predictions(pred_file)
        if not metrics:
            print(f"  ❌ Evaluation failed, skipping...")
            continue
        
        em = metrics.get('em', 0)
        f1 = metrics.get('f1', 0)
        joint_f1 = metrics.get('joint_f1', 0)
        
        # Load performance metrics if available
        metrics_file = pred_file.replace(".json", "_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file) as mf:
                perf = json.load(mf)
                if 'aggregate' in perf:
                    metrics['avg_latency_ms'] = perf['aggregate'].get('avg_latency_ms', 0)
                    metrics['avg_tokens'] = perf['aggregate'].get('avg_tokens_per_example', 0)
        
        print(f"  ✓ EM: {em:.3f} | F1: {f1:.3f} | Joint F1: {joint_f1:.3f}")
        if 'avg_latency_ms' in metrics:
            print(f"    Latency: {metrics['avg_latency_ms']:.1f}ms | Tokens: {metrics['avg_tokens']:.1f}")
        
        result_entry = {
            'evidence_k': evidence_k,
            'sp_k': sp_k,
            'description': desc,
            **metrics
        }
        results.append(result_entry)
        
        if f1 > best_f1:
            best_f1 = f1
            best_config = (evidence_k, sp_k, desc)
            best_metrics = metrics
            print(f"  🌟 New best F1!")
    
    print("\n" + "="*70)
    print("TUNING COMPLETE")
    print("="*70)
    
    if best_config:
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"  evidence_k: {best_config[0]}")
        print(f"  sp_k: {best_config[1]}")
        print(f"  description: {best_config[2]}")
        print(f"\n📊 BEST METRICS:")
        print(f"  EM: {best_metrics['em']:.3f}")
        print(f"  F1: {best_metrics['f1']:.3f}")
        print(f"  Joint EM: {best_metrics['joint_em']:.3f}")
        print(f"  Joint F1: {best_metrics['joint_f1']:.3f}")
        print(f"  SP F1: {best_metrics['sp_f1']:.3f}")
        if 'avg_latency_ms' in best_metrics:
            print(f"  Avg Latency: {best_metrics['avg_latency_ms']:.1f}ms")
            print(f"  Avg Tokens: {best_metrics['avg_tokens']:.1f}")
    
    # Save all results
    results_file = f"results/tuning_{run_id}.json"
    os.makedirs("results", exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'model': LLM_MODEL,
                'limit': LIMIT,
                'device': DEVICE
            },
            'best_config': best_config if best_config else None,
            'best_metrics': best_metrics if best_metrics else None,
            'all_results': results
        }, f, indent=2)
    
    print(f"\n💾 Full results saved to: {results_file}")
    
    # Print comparison table with cost metrics
    print("\n" + "="*100)
    print("COMPARISON TABLE (Quality vs Cost)")
    print("="*100)
    print(f"{'Config':<15} {'EM':<8} {'F1':<8} {'Joint F1':<10} {'SP F1':<8} {'Latency':<12} {'Tokens':<10}")
    print("-" * 100)
    for r in sorted(results, key=lambda x: x['f1'], reverse=True):
        config_str = f"e={r['evidence_k']},sp={r['sp_k']}"
        latency_str = f"{r.get('avg_latency_ms', 0):.1f}ms" if 'avg_latency_ms' in r else "N/A"
        tokens_str = f"{r.get('avg_tokens', 0):.0f}" if 'avg_tokens' in r else "N/A"
        print(f"{config_str:<15} {r['em']:<8.3f} {r['f1']:<8.3f} {r['joint_f1']:<10.3f} {r['sp_f1']:<8.3f} {latency_str:<12} {tokens_str:<10}")
    
    print("="*100)

if __name__ == "__main__":
    main()