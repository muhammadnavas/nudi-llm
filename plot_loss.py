"""
NudiLLM - Training Loss Visualization
Plots training and validation loss curves from the saved log file.

Usage:
  python plot_loss.py
"""

import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def plot_loss():
    log_path = Path("logs/train_log.json")

    if not log_path.exists():
        print("No training log found. Train the model first: python train.py")
        return

    with open(log_path) as f:
        data = json.load(f)

    train_data = data.get("train", [])
    val_data = data.get("val", [])

    if not train_data:
        print("No training data in log.")
        return

    train_steps = [d["step"] for d in train_data]
    train_loss = [d["loss"] for d in train_data]

    if not HAS_MPL:
        # Text-based fallback
        print("\n📊 Training Loss Summary (install matplotlib for plots)")
        print(f"   First loss : {train_loss[0]:.4f}")
        print(f"   Last loss  : {train_loss[-1]:.4f}")
        print(f"   Min loss   : {min(train_loss):.4f}")
        if val_data:
            val_loss = [d["loss"] for d in val_data]
            print(f"   Best val   : {min(val_loss):.4f}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    # Smooth train loss (moving average)
    window = min(50, len(train_loss) // 10 + 1)
    smoothed = []
    for i in range(len(train_loss)):
        start = max(0, i - window)
        smoothed.append(sum(train_loss[start:i+1]) / (i - start + 1))

    ax.plot(train_steps, train_loss, alpha=0.3, color="#e94560", linewidth=0.8, label="Train loss (raw)")
    ax.plot(train_steps, smoothed, color="#e94560", linewidth=2, label="Train loss (smoothed)")

    if val_data:
        val_steps = [d["step"] for d in val_data]
        val_loss = [d["loss"] for d in val_data]
        ax.plot(val_steps, val_loss, "o-", color="#0f3460", markerfacecolor="#4cc9f0",
                linewidth=2, markersize=6, label="Validation loss")

    ax.set_xlabel("Training Steps", color="white", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", color="white", fontsize=12)
    ax.set_title("🇮🇳 NudiLLM — Training Progress", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=11)
    ax.grid(True, alpha=0.2, color="white")

    plt.tight_layout()

    out_path = Path("logs/loss_plot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"✅ Loss plot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    plot_loss()
