# -*- coding: utf-8 -*-
import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
"""
NudiLLM - Step 1: Download & Prepare Kannada Training Data
Downloads Kannada Wikipedia from HuggingFace datasets and extracts clean text.
Target: ~50-100MB of clean Kannada text for training our mini LLM.
"""

import os
import re
import unicodedata
from pathlib import Path
from tqdm import tqdm

# Output paths
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "kannada_corpus.txt"

def normalize_kannada_text(text: str) -> str:
    """
    Clean and normalize Kannada Unicode text.
    - NFC normalization (standard for Indic scripts)
    - Remove URLs, HTML artifacts, excessive whitespace
    - Keep Kannada script (U+0C80–U+0CFF) + punctuation + digits
    """
    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove non-Kannada, non-ASCII printable characters
    # Keep: Kannada block (0C80-0CFF), spaces, newlines, basic punctuation, digits
    def keep_char(c):
        cp = ord(c)
        if 0x0C80 <= cp <= 0x0CFF:  # Kannada block
            return True
        if c in " \n\t.,;:!?()[]{}\"'-–—/\\0123456789":
            return True
        if "a" <= c <= "z" or "A" <= c <= "Z":  # keep English too (bilingual)
            return True
        return False

    text = "".join(c for c in text if keep_char(c))

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def download_kannada_wikipedia():
    """Download Kannada Wikipedia articles from HuggingFace."""
    print("=" * 60)
    print("NudiLLM Data Preparation — Kannada Wikipedia")
    print("=" * 60)
    print("\nDownloading Kannada Wikipedia dataset from HuggingFace...")
    print("(This may take a few minutes on first run)\n")

    try:
        from datasets import load_dataset

        # Kannada Wikipedia — small but high quality
        dataset = load_dataset(
            "wikipedia",
            language="kn",
            date="20231101",
            trust_remote_code=True,
            cache_dir=str(RAW_DIR),
        )
        return dataset["train"]

    except Exception as e:
        print(f"HuggingFace Wikipedia failed: {e}")
        print("Trying fallback: wikimedia/wikipedia...")
        try:
            from datasets import load_dataset

            dataset = load_dataset(
                "wikimedia/wikipedia",
                "20231101.kn",
                cache_dir=str(RAW_DIR),
                trust_remote_code=True,
            )
            return dataset["train"]
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return None


def process_and_save(dataset):
    """Process dataset articles and write to corpus file."""
    print(f"\nProcessing articles → {OUTPUT_FILE}")
    total_chars = 0
    total_articles = 0
    skipped = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for article in tqdm(dataset, desc="Processing articles"):
            text = article.get("text", "")
            if not text:
                skipped += 1
                continue

            # Normalize
            clean = normalize_kannada_text(text)

            # Skip very short articles (stub pages)
            if len(clean) < 200:
                skipped += 1
                continue

            # Write article with separator
            f.write(clean)
            f.write("\n\n")  # Article separator

            total_chars += len(clean)
            total_articles += 1

    size_mb = total_chars / (1024 * 1024)
    print(f"\n✅ Done!")
    print(f"   Articles processed : {total_articles:,}")
    print(f"   Articles skipped   : {skipped:,}")
    print(f"   Total corpus size  : {size_mb:.1f} MB")
    print(f"   Approx. tokens     : ~{int(total_chars / 3):,} (Kannada ~3 chars/token)")
    print(f"   Saved to           : {OUTPUT_FILE}")

    return total_chars


def create_train_val_split(corpus_path: Path, val_ratio: float = 0.05):
    """Split corpus into train and validation sets."""
    print("\nCreating train/val split...")

    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    split_idx = int(len(text) * (1 - val_ratio))
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    train_path = PROCESSED_DIR / "train.txt"
    val_path = PROCESSED_DIR / "val.txt"

    with open(train_path, "w", encoding="utf-8") as f:
        f.write(train_text)
    with open(val_path, "w", encoding="utf-8") as f:
        f.write(val_text)

    print(f"   Train : {len(train_text) / 1024 / 1024:.1f} MB → {train_path}")
    print(f"   Val   : {len(val_text) / 1024 / 1024:.1f} MB → {val_path}")


if __name__ == "__main__":
    print("\n[NudiLLM] Kannada Mini Language Model")
    print("=" * 60)

    dataset = download_kannada_wikipedia()

    if dataset is not None:
        total_chars = process_and_save(dataset)
        if total_chars > 0:
            create_train_val_split(OUTPUT_FILE)
            print("\n✅ Data preparation complete!")
            print("   Next step: python tokenizer/train_tokenizer.py")
    else:
        print("\n❌ Could not download dataset.")
        print("   Check your internet connection and try again.")
        print("   Or manually place a Kannada text file at data/processed/kannada_corpus.txt")
