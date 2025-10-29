# config.py - Central configuration for all experiments
"""
Single source of truth for all model and experiment settings.
Modify this file to change settings across all experiments.
"""

# =============================================================================
# MODEL CONFIGURATION (EDIT HERE)
# =============================================================================

# LLM Configuration
LLM_MODEL = "Qwen/Qwen3-8B"  # Change this to use a different model
LLM_MAX_NEW_TOKENS = 32
LLM_TEMPERATURE = 0.1
LLM_DO_SAMPLE = True
LLM_TOP_P = 0.9

# Embedding Model Configuration
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Device Configuration
DEFAULT_DEVICE = 0,1,2,3  # -1 for CPU, 0+ for GPU

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# HotpotQA Defaults
HOTPOT_EVIDENCE_K = 8  # Number of evidence sentences
HOTPOT_SP_K = 2        # Number of supporting facts
HOTPOT_USE_ANSWER_TYPE = True  # Use answer type detection
HOTPOT_USE_ANSWER_GUIDED_SP = True  # Use answer-guided SP selection

# Multi-hop Retrieval
MULTIHOP_K_PER_HOP = 3
MULTIHOP_NUM_HOPS = 2

# RAG Configuration
RAG_TOP_K = 3  # Default retrieval top-k

# =============================================================================
# PATHS CONFIGURATION
# =============================================================================

# Data paths
DATA_DIR = "data"
RESULTS_DIR = "results"
PREDICTIONS_DIR = "predictions"
RUNS_DIR = "runs"

# Default files
HOTPOT_DEV_JSON = f"{RUNS_DIR}/hotpot_dev_distractor_200.json"
HOTPOT_FULL_DEV = f"{DATA_DIR}/hotpot_dev_distractor_v1.json"

# =============================================================================
# PERFORMANCE TUNING
# =============================================================================

# Cache settings
ENABLE_CACHE = True

# Logging
ENABLE_DETAILED_LOGGING = False

# =============================================================================
# AVAILABLE MODEL PRESETS
# =============================================================================

MODEL_PRESETS = {
    "qwen-8b": {
        "model": "Qwen/Qwen3-8B",
        "max_tokens": 32,
        "temp": 0.1
    },
    "llama-8b": {
        "model": "meta-llama/Meta-Llama-8B-Instruct",
        "max_tokens": 64,
        "temp": 0.2
    }
}

def get_model_config(preset_name=None):
    """
    Get model configuration either from preset or from current settings.
    
    Args:
        preset_name: Optional preset name from MODEL_PRESETS
    
    Returns:
        dict with model configuration
    """
    if preset_name and preset_name in MODEL_PRESETS:
        preset = MODEL_PRESETS[preset_name]
        return {
            "model_name": preset["model"],
            "max_new_tokens": preset["max_tokens"],
            "temperature": preset["temp"],
            "do_sample": True,
            "top_p": 0.9
        }
    
    # Return current configuration
    return {
        "model_name": LLM_MODEL,
        "max_new_tokens": LLM_MAX_NEW_TOKENS,
        "temperature": LLM_TEMPERATURE,
        "do_sample": LLM_DO_SAMPLE,
        "top_p": LLM_TOP_P
    }

def print_config():
    """Print current configuration."""
    print("="*70)
    print("CURRENT CONFIGURATION")
    print("="*70)
    print(f"LLM Model: {LLM_MODEL}")
    print(f"Max Tokens: {LLM_MAX_NEW_TOKENS}")
    print(f"Temperature: {LLM_TEMPERATURE}")
    print(f"Embed Model: {EMBED_MODEL}")
    print(f"Device: {'GPU ' + str(DEFAULT_DEVICE) if DEFAULT_DEVICE >= 0 else 'CPU'}")
    print(f"\nHotpotQA Settings:")
    print(f"  Evidence K: {HOTPOT_EVIDENCE_K}")
    print(f"  SP K: {HOTPOT_SP_K}")
    print(f"  Answer Type Detection: {HOTPOT_USE_ANSWER_TYPE}")
    print(f"  Answer-Guided SP: {HOTPOT_USE_ANSWER_GUIDED_SP}")
    print("="*70)

if __name__ == "__main__":
    print_config()