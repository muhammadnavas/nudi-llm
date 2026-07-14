"""
NudiLLM - Step 3: Training Script
Trains the mini Kannada GPT on your GTX 1650 (4GB VRAM) or Google Colab.

Estimated training time:
  - GTX 1650 (4GB): ~2-5 hours for 3 epochs on Kannada Wikipedia
  - Colab T4 (16GB): ~30-60 minutes for 3 epochs
  - Colab A100 (40GB): ~10-20 minutes for 3 epochs
  - CPU only:         ~3-7 days (not recommended)

Usage:
  python train.py                  # Default config (local GPU)
  python train.py --epochs 5       # More epochs
  python train.py --resume         # Resume from checkpoint
  python train.py --colab          # Colab mode (Drive-backed paths + larger config)
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
# CHECKPOINT_DIR and LOG_DIR are set inside train() based on --version flag


# ── Training Config ────────────────────────────────────────────────────────────
def get_train_config(colab: bool = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Detect BF16 support (A100 / newer Ampere GPUs on Colab Pro)
    supports_bf16 = (
        device == "cuda"
        and torch.cuda.is_bf16_supported()
    )

    if colab:
        # ── Colab T4 (16 GB) / A100 (40 GB) config ──────────────────────────
        # T4 has 4× the VRAM of a GTX 1650, so we can use a larger model and
        # bigger batches. A100 can go even higher, but these settings are safe
        # for both T4 and A100.
        return {
            "vocab_size": 8000,
            "context_length": 1024,    # 2× larger context
            "n_embd": 512,             # wider embeddings
            "n_heads": 8,
            "n_layers": 8,             # deeper model
            "dropout": 0.05,

            # Training
            "batch_size": 16,          # T4 can handle 16 comfortably
            "num_epochs": 3,
            "learning_rate": 1e-4,
            "weight_decay": 0.1,
            "grad_clip": 1.0,
            "warmup_steps": 500,

            # Evaluation
            "eval_interval": 500,
            "eval_steps": 50,
            "save_interval": 500,      # Save more often to survive session drops

            # Device
            "device": device,
            "bf16": supports_bf16,     # BF16 on A100 (better than FP16)
            "fp16": not supports_bf16, # FP16 on T4 (no BF16 support)
        }
    else:
        # ── Local GTX 1650 (4 GB) config ─────────────────────────────────────
        return {
            "vocab_size": 8000,
            "context_length": 512,     # ↑ 256→512: Flash Attention frees enough VRAM
            "n_embd": 384,
            "n_heads": 6,
            "n_layers": 6,
            "dropout": 0.05,           # ↓ 0.1→0.05: less aggressive regularization

            # Training
            # batch_size reduced 8→6 to offset the larger context window.
            # Effective batch = batch_size × ACCUM_STEPS = 6×4 = 24  (defined below)
            "batch_size": 6,
            "num_epochs": 3,
            "learning_rate": 1e-4,     # ↓ 3e-4→1e-4: more stable for longer sequences
            "weight_decay": 0.1,
            "grad_clip": 1.0,          # Gradient clipping
            "warmup_steps": 500,       # ↑ 200→500: smoother warmup for lower LR

            # Evaluation
            "eval_interval": 500,      # Evaluate every N steps
            "eval_steps": 50,          # Steps to average for val loss
            "save_interval": 1000,     # Save checkpoint every N steps

            # Device
            "device": device,
            "bf16": False,             # GTX 1650 Turing has no BF16
            "fp16": True,              # FP16 on GTX 1650
        }


# Gradient accumulation steps.
# Local:  effective batch = 6  × 4 = 24
# Colab:  effective batch = 16 × 2 = 32  (set per-run inside train())
ACCUM_STEPS = 4


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
def train(resume: bool = False, extra_epochs: int = 0, version: str = "v1", colab: bool = False):
    # ── Versioned output directories ───────────────────────────────────────────
    if colab:
        # Save to Google Drive so checkpoints survive session resets
        drive_root = Path("/content/drive/MyDrive/NudiLLM")
        checkpoint_dir = drive_root / "checkpoints" / version
        log_dir = drive_root / "logs" / version
    else:
        checkpoint_dir = Path("checkpoints") / version
        log_dir = Path("logs") / version

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Version] {version}")
    print(f"   Checkpoints : {checkpoint_dir}")
    print(f"   Logs        : {log_dir}")
    if colab:
        print("   [Colab] Checkpoints are saved to Google Drive")

    config = get_train_config(colab=colab)
    device = config["device"]

    print("=" * 60)
    print("NudiLLM --- Kannada Mini Language Model Training")
    print("=" * 60)
    print(f"\n[Device] : {device.upper()}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU   : {gpu_name} ({vram:.1f} GB VRAM)")
    print(f"   BF16  : {config['bf16']}")
    print(f"   FP16  : {config['fp16']}")
    print(f"   Batch : {config['batch_size']}")
    print(f"   Seq   : {config['context_length']}")
    print(f"   Colab : {colab}")
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

    # Gradient Scaler — used for FP16 training. BF16 does not need a scaler.
    use_scaler = config["fp16"] and device == "cuda" and not config["bf16"]
    scaler = GradScaler("cuda", enabled=use_scaler)

    # ── Resume from Checkpoint ─────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    checkpoint_path = checkpoint_dir / "latest.pt"
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

    # On Colab we can afford fewer accumulation steps because batch_size is already large.
    accum_steps = 2 if colab else ACCUM_STEPS

    # steps_per_epoch = optimizer steps per epoch (each step = accum_steps micro-batches)
    steps_per_epoch = len(train_loader) // accum_steps
    total_steps = total_epochs * steps_per_epoch
    config["total_steps"] = total_steps

    # ── Training ───────────────────────────────────────────────────────────────
    print(f"\n[Start] Starting training...")
    print(f"   Epochs         : {total_epochs}")
    print(f"   Steps/epoch    : {steps_per_epoch:,}")
    print(f"   Total steps    : {total_steps:,}")
    print(f"   Accum steps    : {accum_steps}  (effective batch = {config['batch_size'] * accum_steps})")
    print(f"   Eval every     : {config['eval_interval']} steps")
    print(f"   Save every     : {config['save_interval']} steps")
    print()

    log_file = log_dir / "train_log.json"
    model.train()
    t0 = time.time()

    # Convert train_loader to an iterator for manual micro-step control
    train_iter = iter(train_loader)

    for epoch in range(start_epoch, total_epochs):
        print(f"── Epoch {epoch + 1}/{total_epochs} ──────────────────────────────────")
        epoch_loss = 0.0
        steps_this_epoch = 0

        while steps_this_epoch < steps_per_epoch:
            # ── Update learning rate once per optimizer step ────────────────
            lr = get_lr(global_step, config)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # ── Gradient Accumulation ───────────────────────────────────────
            # Accumulate accum_steps micro-batches before updating weights.
            # Each micro-batch contributes loss / accum_steps so the total
            # gradient magnitude equals that of a single batch_size×accum_steps batch.
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            # Determine autocast dtype: BF16 on A100, FP16 on T4/GTX 1650
            if config["bf16"] and device == "cuda":
                amp_dtype = torch.bfloat16
            elif config["fp16"] and device == "cuda":
                amp_dtype = torch.float16
            else:
                amp_dtype = None

            for micro_step in range(accum_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)  # restart iterator
                    x, y = next(train_iter)

                x, y = x.to(device), y.to(device)

                if amp_dtype is not None:
                    with autocast("cuda", dtype=amp_dtype):
                        _, loss = model(x, y)
                    # Divide by accum_steps so gradients average over micro-batches
                    loss = loss / accum_steps
                    scaler.scale(loss).backward()
                else:
                    _, loss = model(x, y)
                    loss = loss / accum_steps
                    loss.backward()

                accum_loss += loss.item()
                steps_this_epoch += 1
                if steps_this_epoch >= steps_per_epoch:
                    break  # don't overshoot the epoch boundary

            # ── Optimizer step (once per accum_steps micro-batches) ─────────
            if use_scaler:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()

            # accum_loss is already divided — multiply back for readable logging
            step_loss = accum_loss * accum_steps
            epoch_loss += step_loss
            global_step += 1

            # ── Logging ────────────────────────────────────────────────────────────
            if global_step % 50 == 0:
                elapsed = time.time() - t0
                steps_done = global_step
                steps_remaining = total_steps - steps_done
                eta_sec = (elapsed / max(steps_done, 1)) * steps_remaining
                eta_min = eta_sec / 60

                print(
                    f"  Step {global_step:5d} | "
                    f"Loss: {step_loss:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"ETA: {eta_min:.0f}m"
                )
                train_losses.append({"step": global_step, "loss": step_loss})
            # ── Validation ────────────────────────────────────────────────────
            if global_step % config["eval_interval"] == 0:
                val_loss = evaluate(model, val_loader, config)
                print(f"\n  [Val] Val Loss: {val_loss:.4f} (Train: {step_loss:.4f})")
                val_losses.append({"step": global_step, "loss": val_loss})

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path = checkpoint_dir / "best.pt"
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
    final_path = checkpoint_dir / "final.pt"
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
    print(f"   Best model    : {checkpoint_dir / 'best.pt'}")
    print()
    print("   Next step: python inference.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NudiLLM mini Kannada LLM")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=0, help="Extra epochs to train")
    parser.add_argument("--version", type=str, default="v1",
                        help="Training version tag (e.g. v1, v2). "
                             "Checkpoints saved to checkpoints/<version>/")
    parser.add_argument("--colab", action="store_true",
                        help="Colab mode: larger config + checkpoints saved to "
                             "/content/drive/MyDrive/NudiLLM/")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available. Training on CPU will be very slow.")
        if args.colab:
            print("   In Colab: Runtime → Change runtime type → GPU (T4)")
        else:
            print("   Make sure CUDA drivers and PyTorch CUDA are installed.")
            print("   Install: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        ans = input("\nContinue on CPU anyway? (y/N): ")
        if ans.lower() != "y":
            sys.exit(0)

    train(resume=args.resume, extra_epochs=args.epochs, version=args.version, colab=args.colab)
