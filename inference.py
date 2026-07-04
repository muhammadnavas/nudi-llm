"""
NudiLLM - Step 4: Inference — Generate Kannada Text
Loads a trained checkpoint and generates Kannada text interactively.

Usage:
  python inference.py                          # Interactive chat
  python inference.py --prompt "ಕನ್ನಡ"         # With starting prompt
  python inference.py --model checkpoints/best.pt
"""

import sys
import argparse
from pathlib import Path

import torch
import sentencepiece as spm

sys.path.insert(0, str(Path(__file__).parent))
from model.nudi import NudiLLM, NudiConfig


def load_model(checkpoint_path: str, device: str):
    """Load a trained NudiLLM model from a checkpoint."""
    print(f"Loading model from {checkpoint_path}...")

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_dict = ckpt["config"]

    model_config = NudiConfig(**cfg_dict)
    model = NudiLLM(model_config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    params = model.count_parameters()
    print(f"✅ Model loaded! Parameters: {params:,} (~{params/1e6:.1f}M)")
    return model, model_config


def generate_text(
    model: NudiLLM,
    sp: spm.SentencePieceProcessor,
    prompt: str,
    device: str,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
) -> str:
    """Generate Kannada text continuing from a prompt."""
    # Encode prompt
    token_ids = sp.encode(prompt)
    if not token_ids:
        token_ids = [sp.bos_id()]

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    # Decode — only the newly generated tokens
    new_token_ids = output_ids[0, len(token_ids):].tolist()
    generated = sp.decode(new_token_ids)

    return generated


def interactive_mode(model, sp, device):
    """Interactive text generation loop."""
    print("\n" + "=" * 60)
    print("🇮🇳 NudiLLM — Kannada Text Generator (Interactive Mode)")
    print("=" * 60)
    print("Type a Kannada word/phrase to continue, or type 'quit' to exit.")
    print("Commands: :temp <0.1-2.0>  :tokens <N>  :topk <N>")
    print("-" * 60)

    temperature = 0.8
    max_tokens = 150
    top_k = 50

    while True:
        try:
            prompt = input("\n📝 Prompt (ಕನ್ನಡ): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! ನಮಸ್ಕಾರ!")
            break

        if prompt.lower() == "quit":
            print("ನಮಸ್ಕಾರ! Goodbye!")
            break

        # Handle commands
        if prompt.startswith(":temp "):
            try:
                temperature = float(prompt.split()[1])
                print(f"   Temperature set to {temperature}")
            except:
                print("   Usage: :temp 0.8")
            continue
        elif prompt.startswith(":tokens "):
            try:
                max_tokens = int(prompt.split()[1])
                print(f"   Max tokens set to {max_tokens}")
            except:
                print("   Usage: :tokens 100")
            continue
        elif prompt.startswith(":topk "):
            try:
                top_k = int(prompt.split()[1])
                print(f"   Top-K set to {top_k}")
            except:
                print("   Usage: :topk 50")
            continue

        if not prompt:
            prompt = "ಕನ್ನಡ"  # Default: start with "Kannada"

        print(f"\n🤖 Generating...\n")
        generated = generate_text(
            model, sp, prompt, device,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        print(f"{'─' * 50}")
        print(f"[Prompt] {prompt}")
        print(f"[Generated] {generated}")
        print(f"{'─' * 50}")

        # Save to file for better font rendering
        with open("generated.txt", "w", encoding="utf-8") as f:
            f.write(f"Prompt: {prompt}\n\nGenerated:\n{generated}\n")
        print("💡 (Saved to generated.txt because terminal fonts often break Kannada characters!)")

        # Show token count
        total_tokens = len(sp.encode(prompt + generated))
        print(f"(~{total_tokens} tokens, temp={temperature}, top_k={top_k})\n")


def main():
    parser = argparse.ArgumentParser(description="NudiLLM Kannada Text Generation")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Starting prompt (Kannada text). If not given, starts interactive mode.",
    )
    parser.add_argument("--tokens", type=int, default=150, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K filtering")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P (nucleus) filtering")
    parser.add_argument("--device", type=str, default=None, help="Force device (e.g., 'cpu' or 'cuda')")
    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Check checkpoint
    if not Path(args.model).exists():
        print(f"\n❌ Checkpoint not found: {args.model}")
        print("   Run training first: python train.py")
        sys.exit(1)

    # Check tokenizer
    tokenizer_path = "tokenizer/kannada_bpe.model"
    if not Path(tokenizer_path).exists():
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        sys.exit(1)

    # Load model and tokenizer
    model, _ = load_model(args.model, device)
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)

    if args.prompt:
        # Single generation
        print(f"\n📝 Prompt: {args.prompt}")
        print("🤖 Generating...\n")
        generated = generate_text(
            model, sp, args.prompt, device,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(f"{'─' * 50}")
        print(f"[Prompt]    {args.prompt}")
        print(f"[Generated] {generated}")
        print(f"{'─' * 50}")
    else:
        # Interactive mode
        interactive_mode(model, sp, device)


if __name__ == "__main__":
    main()
