"""
NudiLLM - Quick Sanity Test
Verifies the model architecture works correctly BEFORE full training.
Run this first to catch any errors quickly (< 30 seconds).

Usage:
  python test_model.py
"""

import sys
import torch
from pathlib import Path

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from model.nudi import NudiLLM, NudiConfig


def run_tests():
    print("=" * 60)
    print("🔧 NudiLLM Model Sanity Tests")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {vram:.1f} GB")

    # ── Test 1: Model Construction ────────────────────────────────────────────
    print("\n[1/5] Model construction...")
    config = NudiConfig()
    model = NudiLLM(config).to(device)
    params = model.count_parameters()
    print(f"  ✅ Model created: {params:,} parameters ({params/1e6:.1f}M)")
    print(f"  {model}")

    # ── Test 2: Forward Pass ──────────────────────────────────────────────────
    print("\n[2/5] Forward pass (batch=4, seq=64)...")
    B, T = 4, 64
    x = torch.randint(0, config.vocab_size, (B, T)).to(device)
    y = torch.randint(0, config.vocab_size, (B, T)).to(device)

    with torch.no_grad():
        logits, loss = model(x, y)

    assert logits.shape == (B, T, config.vocab_size), f"Wrong logits shape: {logits.shape}"
    assert loss is not None
    print(f"  ✅ Logits shape: {tuple(logits.shape)}")
    print(f"  ✅ Loss: {loss.item():.4f} (should be ~ln({config.vocab_size}) = {__import__('math').log(config.vocab_size):.2f} at init)")

    # ── Test 3: FP16 Forward Pass ─────────────────────────────────────────────
    if device == "cuda":
        print("\n[3/5] FP16 forward pass...")
        from torch.cuda.amp import autocast
        with autocast(dtype=torch.float16):
            logits_fp16, loss_fp16 = model(x, y)
        print(f"  ✅ FP16 logits shape: {tuple(logits_fp16.shape)}")
        print(f"  ✅ FP16 loss: {loss_fp16.item():.4f}")
    else:
        print("\n[3/5] FP16 test (skipped — no CUDA)")

    # ── Test 4: Generation ────────────────────────────────────────────────────
    print("\n[4/5] Text generation (random tokens)...")
    prompt = torch.randint(0, config.vocab_size, (1, 5)).to(device)
    with torch.no_grad():
        output = model.generate(prompt, max_new_tokens=20, temperature=1.0)
    assert output.shape[1] > 5
    print(f"  ✅ Generated {output.shape[1] - 5} new tokens")
    print(f"  ✅ Output shape: {tuple(output.shape)}")

    # ── Test 5: VRAM Usage ────────────────────────────────────────────────────
    if device == "cuda":
        print("\n[5/5] VRAM estimation (batch=8, seq=256)...")
        torch.cuda.reset_peak_memory_stats()
        B, T = 8, 256
        x_big = torch.randint(0, config.vocab_size, (B, T)).to(device)
        y_big = torch.randint(0, config.vocab_size, (B, T)).to(device)

        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad()

        with autocast(dtype=torch.float16):
            _, loss = model(x_big, y_big)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        peak_vram = torch.cuda.max_memory_allocated() / 1024**3
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        pct = (peak_vram / total_vram) * 100
        print(f"  ✅ Peak VRAM: {peak_vram:.2f} GB / {total_vram:.1f} GB ({pct:.0f}%)")
        if pct < 85:
            print(f"  ✅ Within safe limit for GTX 1650!")
        else:
            print(f"  ⚠️  High VRAM usage. Consider reducing batch_size to 4.")
    else:
        print("\n[5/5] VRAM test (skipped — no CUDA)")

    print("\n" + "=" * 60)
    print("✅ All tests passed! Model is ready for training.")
    print("   Next: python data/prepare_data.py")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
