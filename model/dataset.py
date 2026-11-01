"""
NudiLLM - Dataset loader for training.
Reads tokenized Kannada text and serves batches of (input, target) pairs.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sentencepiece as spm
import numpy as np


class KannadaDataset(Dataset):
    """
    Token-level dataset for language model training.
    
    Reads the entire corpus into memory as token ids, then serves
    sliding-window chunks of length `context_length`.
    
    For a sequence: [t0, t1, t2, ..., tN]
    Input:  [t0, t1, ..., t_{L-1}]
    Target: [t1, t2, ..., t_L]     ← shifted by 1 (next-token prediction)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        context_length: int = 256,
        split: str = "train",
    ):
        self.context_length = context_length
        self.split = split

        # Load tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(tokenizer_path)

        # Load and tokenize corpus (with binary cache for fast restarts)
        data_path = Path(data_path)
        cache_path = data_path.with_suffix(".tokens.npy")

        if cache_path.exists():
            print(f"  Loading cached tokens from {cache_path} ...")
            self.tokens = np.load(str(cache_path))
        else:
            print(f"  Loading {split} data from {data_path}...")
            with open(data_path, "r", encoding="utf-8") as f:
                text = f.read()

            print(f"  Tokenizing {len(text) / 1024 / 1024:.1f}MB of text...")
            chunk_size = 100_000
            all_ids = []
            for i in range(0, len(text), chunk_size):
                chunk = text[i : i + chunk_size]
                ids = self.sp.encode(chunk)
                all_ids.extend(ids)

            self.tokens = np.array(all_ids, dtype=np.int32)

            # Save cache so next run is instant
            np.save(str(cache_path), self.tokens)
            print(f"  Token cache saved to {cache_path}")
        print(f"  Total tokens: {len(self.tokens):,}")
        print(f"  Chunks of {context_length}: {len(self):,}")

    def __len__(self):
        # Number of non-overlapping chunks
        return (len(self.tokens) - 1) // self.context_length

    def __getitem__(self, idx):
        start = idx * self.context_length
        end = start + self.context_length

        # Input: tokens[start:end], Target: tokens[start+1:end+1]
        x = torch.from_numpy(self.tokens[start:end].astype(np.int64))
        y = torch.from_numpy(self.tokens[start + 1 : end + 1].astype(np.int64))
        return x, y


def create_dataloaders(
    tokenizer_path: str,
    context_length: int = 256,
    batch_size: int = 8,
    num_workers: int = 0,  # 0 for Windows compatibility
):
    """Create train and validation DataLoaders."""
    train_dataset = KannadaDataset(
        data_path="data/processed/train.txt",
        tokenizer_path=tokenizer_path,
        context_length=context_length,
        split="train",
    )
    val_dataset = KannadaDataset(
        data_path="data/processed/val.txt",
        tokenizer_path=tokenizer_path,
        context_length=context_length,
        split="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
