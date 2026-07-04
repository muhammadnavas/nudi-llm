"""
NudiLLM - Step 3: Training Script
Trains the mini Kannada GPT on your GTX 1650 (4GB VRAM).

Estimated training time:
  - GTX 1650 (4GB): ~2-5 hours for 3 epochs on Kannada Wikipedia
  - CPU only:       ~3-7 days (not recommended)

Usage:
  python train.py                  # Default config
  python train.py --epochs 5       # More epochs
  python train.py --resume         # Resume from checkpoint
"""

import os
import sys
import time
import math
import json
import argparse
from pathlib import Path

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

# Project imports
sys.path.insert(0, str(Path(__file__).parent))
from model.nudi import NudiLLM, NudiConfig
from model.dataset import create_dataloaders

# ── Paths ─────────────────────────────────────────────────────────────────────
TOKENIZER_PATH = "tokenizer/kannada_bpe.model"
CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


# ── Training Config ────────────────────────────────────────────────────────────
def get_train_config():
    return {
        # Model (matches NudiConfig defaults — ~20M params)
        "vocab_size": 8000,
        "context_length": 256,
        "n_embd": 384,
        "n_heads": 6,
        "n_layers": 6,
        "dropout": 0.1,

        # Training
        "batch_size": 8,           # Safe for 4GB VRAM with FP16
        "num_epochs": 3,
        "learning_rate": 3e-4,     # AdamW LR
        "weight_decay": 0.1,
        "grad_clip": 1.0,          # Gradient clipping
        "warmup_steps": 200,       # Linear warmup steps

        # Evaluation
        "eval_interval": 500,      # Evaluate every N steps
        "eval_steps": 50,          # Steps to average for val loss
        "save_interval": 1000,     # Save checkpoint every N steps

        # Device
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "fp16": True,              # FP16 on GTX 1650 (no BF16 support)
    }


# ── Learning Rate Schedule ─────────────────────────────────────────────────────
def get_lr(step: int, config: dict) -> float:
    """
    Cosine decay with linear warmup.
    - Warmup: linear increase from 0 → max_lr
    - Decay: cosine annealing to max_lr * 0.1
    """
    max_lr = config["learning_rate"]
    min_lr = max_lr * 0.1
    warmup = config["warmup_steps"]
    total_steps = config.get("total_steps", 10000)

    if step < warmup:
        return max_lr * step / warmup
    if step > total_steps:
        return min_lr

    decay_ratio = (step - warmup) / (total_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ── Evaluation ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, val_loader, config, scaler=None):
    """Compute validation loss."""
    model.eval()
    device = config["device"]
    total_loss = 0.0
    steps = 0

    for x, y in val_loader:
        if steps >= config["eval_steps"]:
            break
        x, y = x.to(device), y.to(device)

        if config["fp16"] and device == "cuda":
            with autocast("cuda", dtype=torch.float16):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)

        total_loss += loss.item()
        steps += 1

    model.train()
    return total_loss / max(steps, 1)


# ── Training Loop ──────────────────────────────────────────────────────────────
def train(resume: bool = False, extra_epochs: int = 0):
    config = get_train_config()
    device = config["device"]

    print("=" * 60)
    print("NudiLLM --- Kannada Mini Language Model Training")
    print("=" * 60)
    print(f"\n[Device] : {device.upper()}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU   : {gpu_name} ({vram:.1f} GB VRAM)")
    print(f"   FP16  : {config['fp16']}")
    print(f"   Batch : {config['batch_size']}")
    print(f"   Seq   : {config['context_length']}")
    print()

    # Check tokenizer
    if not Path(TOKENIZER_PATH).exists():
        print("❌ Tokenizer not found!")
        print("   Run: python tokenizer/train_tokenizer.py")
        return

    # ── Build Model ────────────────────────────────────────────────────────────
    model_config = NudiConfig(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        n_embd=config["n_embd"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
    )
    model = NudiLLM(model_config).to(device)
    print(model)

    # ── Load Data ──────────────────────────────────────────────────────────────
    print("\n[Data] Loading datasets...")
    train_loader, val_loader = create_dataloaders(
        tokenizer_path=TOKENIZER_PATH,
        context_length=config["context_length"],
        batch_size=config["batch_size"],
    )

    # ── Optimizer ──────────────────────────────────────────────────────────────
    # Separate weight decay: apply to weights, NOT to biases/norms
    decay_params = [p for n, p in model.named_parameters()
                    if p.dim() >= 2 and p.requires_grad]
    nodecay_params = [p for n, p in model.named_parameters()
                      if p.dim() < 2 and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config["weight_decay"]},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=config["learning_rate"],
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # FP16 Gradient Scaler (GTX 1650 doesn't support BF16)
    scaler = GradScaler("cuda", enabled=(config["fp16"] and device == "cuda"))

    # ── Resume from Checkpoint ─────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    checkpoint_path = CHECKPOINT_DIR / "latest.pt"
    if resume and checkpoint_path.exists():
        print(f"\n[Resume] Resuming from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        train_losses = ckpt.get("train_losses", [])
        val_losses = ckpt.get("val_losses", [])
        print(f"   Resumed at epoch {start_epoch}, step {global_step}")

    total_epochs = config["num_epochs"] + extra_epochs
    steps_per_epoch = len(train_loader)
    total_steps = total_epochs * steps_per_epoch
    config["total_steps"] = total_steps

    # ── Training ───────────────────────────────────────────────────────────────
    print(f"\n[Start] Starting training...")
    print(f"   Epochs      : {total_epochs}")
    print(f"   Steps/epoch : {steps_per_epoch:,}")
    print(f"   Total steps : {total_steps:,}")
    print(f"   Eval every  : {config['eval_interval']} steps")
    print(f"   Save every  : {config['save_interval']} steps")
    print()

    log_file = LOG_DIR / "train_log.json"
    model.train()
    t0 = time.time()

    for epoch in range(start_epoch, total_epochs):
        print(f"── Epoch {epoch + 1}/{total_epochs} ──────────────────────────────────")
        epoch_loss = 0.0

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Update learning rate
            lr = get_lr(global_step, config)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Forward + backward with FP16
            optimizer.zero_grad(set_to_none=True)

            if config["fp16"] and device == "cuda":
                with autocast("cuda", dtype=torch.float16):
                    _, loss = model(x, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                _, loss = model(x, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            # ── Logging ────────────────────────────────────────────────────────
            if global_step % 50 == 0:
                elapsed = time.time() - t0
                steps_done = global_step
                steps_remaining = total_steps - steps_done
                eta_sec = (elapsed / max(steps_done, 1)) * steps_remaining
                eta_min = eta_sec / 60

                print(
                    f"  Step {global_step:5d} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {lr:.2e} | "
                    f"ETA: {eta_min:.0f}m"
                )
                train_losses.append({"step": global_step, "loss": loss.item()})
            # ── Validation ────────────────────────────────────────────────────
            if global_step % config["eval_interval"] == 0:
                val_loss = evaluate(model, val_loader, config)
                print(f"\n  [Val] Val Loss: {val_loss:.4f} (Train: {loss.item():.4f})")
                val_losses.append({"step": global_step, "loss": val_loss})

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = CHECKPOINT_DIR / "best.pt"
                    torch.save({
                        "model": model.state_dict(),
                        "config": model_config.__dict__,
                        "val_loss": val_loss,
                        "global_step": global_step,
                    }, best_path)
                    print(f"  [Best] New best! Saved to {best_path}")
                print()

            # ── Checkpoint ────────────────────────────────────────────────────
            if global_step % config["save_interval"] == 0:
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": model_config.__dict__,
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                    "train_losses": train_losses[-500:],  # Keep last 500
                    "val_losses": val_losses,
                }, checkpoint_path)
                print(f"  [Saved] Checkpoint (step {global_step})")

        avg_epoch_loss = epoch_loss / steps_per_epoch
        print(f"\n  Epoch {epoch + 1} complete | Avg Loss: {avg_epoch_loss:.4f}\n")

    # ── Final Save ─────────────────────────────────────────────────────────────
    total_time = (time.time() - t0) / 60
    final_path = CHECKPOINT_DIR / "final.pt"
    torch.save({
        "model": model.state_dict(),
        "config": model_config.__dict__,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
    }, final_path)

    # Save loss log
    with open(log_file, "w") as f:
        json.dump({"train": train_losses, "val": val_losses}, f)

    print("=" * 60)
    print("Training Complete!")
    print(f"   Total time    : {total_time:.1f} minutes")
    print(f"   Best val loss : {best_val_loss:.4f}")
    print(f"   Final model   : {final_path}")
    print(f"   Best model    : {CHECKPOINT_DIR / 'best.pt'}")
    print()
    print("   Next step: python inference.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NudiLLM mini Kannada LLM")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=0, help="Extra epochs to train")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available. Training on CPU will be very slow.")
        print("   Make sure CUDA drivers and PyTorch CUDA are installed.")
        print("   Install: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        ans = input("\nContinue on CPU anyway? (y/N): ")
        if ans.lower() != "y":
            sys.exit(0)

    train(resume=args.resume, extra_epochs=args.epochs)
