import torch
import sys
from pathlib import Path

# Fix Windows encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def check_status():
    ckpt_path = Path("checkpoints/latest.pt")
    if not ckpt_path.exists():
        print("No checkpoint found yet! Wait for step 1,000.")
        return

    try:
        # Load the checkpoint metadata (without loading to GPU)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        epoch = ckpt.get("epoch", 0)
        step = ckpt.get("global_step", 0)
        loss = ckpt.get("best_val_loss", 0)
        
        print("=" * 40)
        print("📊 NudiLLM Training Status")
        print("=" * 40)
        print(f"  Current Epoch  : {epoch}")
        print(f"  Global Step    : {step:,}")
        print(f"  Best Val Loss  : {loss:.4f}")
        print("=" * 40)
        
    except Exception as e:
        print(f"Error reading checkpoint: {e}")

if __name__ == "__main__":
    check_status()
