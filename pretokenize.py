"""
NudiLLM - Pre-tokenization Script
Converts raw text files to binary .npy token arrays ONCE.
Training then loads them instantly (seconds, not minutes).

Usage:
  python pretokenize.py
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
import numpy as np
import sentencepiece as spm
import time

TOKENIZER_PATH = "tokenizer/kannada_bpe.model"
FILES = [
    "data/processed/train.txt",
    "data/processed/val.txt",
]

def tokenize_file(text_path: str, sp: spm.SentencePieceProcessor):
    text_path = Path(text_path)
    npy_path  = text_path.with_suffix(".tokens.npy")

    if npy_path.exists():
        tokens = np.load(str(npy_path))
        print(f"  [SKIP] {npy_path.name} already exists ({len(tokens):,} tokens)")
        return

    print(f"  Reading {text_path.name} ({text_path.stat().st_size/1024/1024:.1f} MB)...")
    t0 = time.time()

    # Read full file
    text = text_path.read_text(encoding="utf-8")

    # Tokenize in larger chunks (1M chars) for speed + fewer function-call overheads
    chunk_size  = 1_000_000  # 1M chars per chunk (much faster than 100K)
    all_ids     = []
    total_chars = len(text)
    n_chunks    = (total_chars + chunk_size - 1) // chunk_size

    print(f"  Tokenizing {total_chars/1e6:.1f}M chars in {n_chunks} chunks...")
    for i in range(n_chunks):
        start = i * chunk_size
        chunk = text[start : start + chunk_size]
        ids   = sp.encode(chunk)
        all_ids.extend(ids)
        pct = (i + 1) / n_chunks * 100
        elapsed = time.time() - t0
        rate    = (start + len(chunk)) / elapsed / 1024  # KB/s (chars)
        print(f"  {pct:5.1f}%  chunk {i+1}/{n_chunks}  "
              f"tokens so far: {len(all_ids):,}  speed: {rate:.0f} Kchar/s",
              end="\r")

    print()
    tokens = np.array(all_ids, dtype=np.int32)
    np.save(str(npy_path), tokens)

    elapsed = time.time() - t0
    print(f"  Saved {npy_path.name}: {len(tokens):,} tokens  "
          f"({npy_path.stat().st_size/1024/1024:.1f} MB)  "
          f"in {elapsed:.0f}s")

def main():
    print("=" * 60)
    print("NudiLLM Pre-tokenizer — building .npy token cache")
    print("=" * 60)

    if not Path(TOKENIZER_PATH).exists():
        print(f"Tokenizer not found: {TOKENIZER_PATH}")
        sys.exit(1)

    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER_PATH)
    print(f"Tokenizer loaded: vocab_size={sp.get_piece_size()}\n")

    t_total = time.time()
    for f in FILES:
        if not Path(f).exists():
            print(f"  [WARN] {f} not found, skipping.")
            continue
        tokenize_file(f, sp)
        print()

    print(f"Done! Total time: {time.time()-t_total:.0f}s")
    print("Now run: python train.py")

if __name__ == "__main__":
    main()
