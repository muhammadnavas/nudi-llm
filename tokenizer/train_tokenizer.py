"""
KALLM - Step 2: Train a Kannada BPE Tokenizer using SentencePiece
Trains a Byte-Pair Encoding tokenizer specifically on Kannada text.
Vocab size: 8000 (efficient for 4GB VRAM constraint)
"""

import os
import sys
import json
from pathlib import Path

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROCESSED_DIR = Path("data/processed")
TOKENIZER_DIR = Path("tokenizer")
TOKENIZER_DIR.mkdir(exist_ok=True)

CORPUS_FILE = PROCESSED_DIR / "kannada_corpus.txt"
TOKENIZER_PREFIX = str(TOKENIZER_DIR / "kannada_bpe")
VOCAB_SIZE = 8000  # Tuned for small model + 4GB VRAM


def train_tokenizer():
    """Train a SentencePiece BPE tokenizer on the Kannada corpus."""
    import sentencepiece as spm

    print("=" * 60)
    print("KALLM — Training Kannada BPE Tokenizer")
    print("=" * 60)

    if not CORPUS_FILE.exists():
        print(f"❌ Corpus not found at {CORPUS_FILE}")
        print("   Run: python data/prepare_data.py first")
        return False

    corpus_size = CORPUS_FILE.stat().st_size / (1024 * 1024)
    print(f"\n📄 Corpus: {CORPUS_FILE} ({corpus_size:.1f} MB)")
    print(f"📝 Vocabulary size: {VOCAB_SIZE}")
    print(f"🔤 Algorithm: BPE (Byte-Pair Encoding)")
    print("\nTraining tokenizer... (takes ~1-3 minutes)")

    spm.SentencePieceTrainer.train(
        input=str(CORPUS_FILE),
        model_prefix=TOKENIZER_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",             # BPE works well for Kannada morphology
        character_coverage=0.9995,    # High coverage for Kannada Unicode chars
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        # normalization_rule_name="nfc" not needed — NFC applied in data prep
        input_sentence_size=5_000_000,  # Max sentences to train on
        shuffle_input_sentence=True,
        num_threads=4,
    )

    print(f"\n✅ Tokenizer saved to: {TOKENIZER_PREFIX}.model")
    return True


def evaluate_tokenizer():
    """Test the tokenizer on sample Kannada sentences."""
    import sentencepiece as spm

    model_path = TOKENIZER_PREFIX + ".model"
    if not Path(model_path).exists():
        print("Tokenizer model not found.")
        return

    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)

    test_sentences = [
        "ನಾನು ಕನ್ನಡ ಕಲಿಯುತ್ತಿದ್ದೇನೆ.",           # I am learning Kannada
        "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ.",             # Bengaluru is the capital of Karnataka
        "ಕನ್ನಡ ಒಂದು ಸುಂದರ ಭಾಷೆ.",                  # Kannada is a beautiful language
        "ಇಂದು ಮಳೆ ಬರುತ್ತದೆ.",                      # It will rain today
        "ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆ ತಂತ್ರಜ್ಞಾನ ಬೆಳೆಯುತ್ತಿದೆ.",   # AI technology is growing
    ]

    print("\n" + "=" * 60)
    print("Tokenizer Evaluation — Fertility Test")
    print("=" * 60)
    print(f"{'Sentence':<50} {'Tokens':>6} {'Words':>5} {'Fertility':>9}")
    print("-" * 75)

    total_tokens = 0
    total_words = 0

    for sent in test_sentences:
        tokens = sp.encode(sent, out_type=str)
        words = sent.split()
        fertility = len(tokens) / max(len(words), 1)
        total_tokens += len(tokens)
        total_words += len(words)

        # Show a truncated version
        display = sent[:45] + "..." if len(sent) > 48 else sent
        print(f"{display:<50} {len(tokens):>6} {len(words):>5} {fertility:>9.2f}")

    avg_fertility = total_tokens / max(total_words, 1)
    print("-" * 75)
    print(f"{'AVERAGE FERTILITY':<50} {total_tokens:>6} {total_words:>5} {avg_fertility:>9.2f}")
    print()
    print("(Lower fertility = more efficient tokenization)")
    print("(Good Kannada tokenizer: fertility < 3.0)")

    # Show a sample tokenization
    sample = "ಕನ್ನಡ ಭಾಷೆ ಬಹಳ ಸಮೃದ್ಧವಾಗಿದೆ"
    tokens = sp.encode(sample, out_type=str)
    ids = sp.encode(sample)
    print(f"\nSample: '{sample}'")
    print(f"Tokens: {tokens}")
    print(f"IDs   : {ids}")

    # Save tokenizer config
    config = {
        "vocab_size": sp.get_piece_size(),
        "model_type": "bpe",
        "pad_id": sp.pad_id(),
        "unk_id": sp.unk_id(),
        "bos_id": sp.bos_id(),
        "eos_id": sp.eos_id(),
        "model_file": "tokenizer/kannada_bpe.model",
    }
    config_path = TOKENIZER_DIR / "tokenizer_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"\n✅ Config saved to: {config_path}")


if __name__ == "__main__":
    print("\n🇮🇳 KALLM — Kannada Tokenizer Training")

    success = train_tokenizer()
    if success:
        evaluate_tokenizer()
        print("\n✅ Tokenizer training complete!")
        print("   Next step: python train.py")
