#!/usr/bin/env python3
"""
Quick script to switch between model presets in config.py
Usage: python switch_model.py qwen-8b
"""
import sys
import re
from pathlib import Path

PRESETS = {
    "qwen-8b": ("Qwen/Qwen3-8B", 32, 0.1),
    "llama-8b": ("meta-llama/Meta-Llama-8B-Instruct", 64, 0.2),
}

def update_config(preset_name):
    """Update config.py with the specified preset."""
    if preset_name not in PRESETS:
        print(f"❌ Unknown preset: {preset_name}")
        print(f"Available presets: {', '.join(PRESETS.keys())}")
        return False
    
    model, max_tokens, temp = PRESETS[preset_name]
    
    config_path = Path("config.py")
    if not config_path.exists():
        print("❌ config.py not found in current directory")
        return False
    
    # Read current config
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Replace values
    content = re.sub(
        r'LLM_MODEL = "[^"]*"',
        f'LLM_MODEL = "{model}"',
        content
    )
    content = re.sub(
        r'LLM_MAX_NEW_TOKENS = \d+',
        f'LLM_MAX_NEW_TOKENS = {max_tokens}',
        content
    )
    content = re.sub(
        r'LLM_TEMPERATURE = [\d.]+',
        f'LLM_TEMPERATURE = {temp}',
        content
    )
    
    # Write back
    with open(config_path, 'w') as f:
        f.write(content)
    
    print("✅ Config updated successfully!")
    print(f"   Model: {model}")
    print(f"   Max Tokens: {max_tokens}")
    print(f"   Temperature: {temp}")
    return True

def show_current():
    """Show current config settings."""
    try:
        import config
        print("\n" + "="*70)
        print("CURRENT CONFIGURATION")
        print("="*70)
        print(f"Model: {config.LLM_MODEL}")
        print(f"Max Tokens: {config.LLM_MAX_NEW_TOKENS}")
        print(f"Temperature: {config.LLM_TEMPERATURE}")
        print(f"Device: {'GPU ' + str(config.DEFAULT_DEVICE) if config.DEFAULT_DEVICE >= 0 else 'CPU'}")
        print(f"\nHotpotQA Settings:")
        print(f"  Evidence K: {config.HOTPOT_EVIDENCE_K}")
        print(f"  SP K: {config.HOTPOT_SP_K}")
        print(f"  Answer Type: {config.HOTPOT_USE_ANSWER_TYPE}")
        print(f"  Answer-Guided SP: {config.HOTPOT_USE_ANSWER_GUIDED_SP}")
        print("="*70)
    except ImportError:
        print("❌ Cannot import config.py")

def main():
    if len(sys.argv) < 2:
        print("Usage: python switch_model.py <preset>")
        print("\nAvailable presets:")
        for name, (model, tokens, temp) in PRESETS.items():
            print(f"  {name:<15} → {model} (tokens={tokens}, temp={temp})")
        print("\nExamples:")
        print("  python switch_model.py qwen-8b")
        print("  python switch_model.py qwen-2.5-7b")
        print("\nTo see current config:")
        print("  python switch_model.py --show")
        return
    
    if sys.argv[1] in ["--show", "-s"]:
        show_current()
        return
    
    preset = sys.argv[1]
    if update_config(preset):
        print("\n💡 Tip: Run any experiment now to use the new model!")
        print("   Example: python experiments/hotpot_dev_predict_distractor.py ...")

if __name__ == "__main__":
    main()