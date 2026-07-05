"""
NudiLLM v1 - Instruction Fine-Tuning
Turns the base autocomplete model into a chatbot.
"""

import os
import sys
import json
import time
from pathlib import Path

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

# Project imports
from model.nudi import NudiLLM, NudiConfig

# ── Paths ─────────────────────────────────────────────────────────────────────
TOKENIZER_PATH = "tokenizer/kannada_bpe.model"
CHECKPOINT_DIR = Path("checkpoints")
INSTRUCT_DATA_PATH = Path("data/instruct_kannada.json")

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "batch_size": 4,          # Small batch size for small dataset
    "epochs": 150,            # 150 passes over the data to memorize the format
    "learning_rate": 1e-4,    # Increased LR slightly to force updates
    "context_length": 256,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ── Dataset ───────────────────────────────────────────────────────────────────
class InstructDataset(Dataset):
    def __init__(self, data_path, sp, context_length):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.samples = []
        self.context_length = context_length
        self.pad_id = sp.pad_id()
        self.eos_id = sp.eos_id()

        for item in data:
            # Format: <|user|>\n[PROMPT]\n<|ai|>\n[RESPONSE]<eos>
            # (We use raw strings here since BPE will just tokenize it normally)
            text = f"<|user|>\n{item['prompt']}\n<|ai|>\n{item['response']}"
            
            # Encode
            tokens = sp.encode(text)
            tokens.append(self.eos_id) # Add End of Sequence
            
            # Truncate if too long
            if len(tokens) > context_length:
                tokens = tokens[:context_length]
                
            self.samples.append(tokens)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        
        # Pad sequence
        x = tokens[:-1]
        y = tokens[1:]
        
        pad_len = self.context_length - len(x)
        if pad_len > 0:
            x = x + [self.pad_id] * pad_len
            y = y + [self.pad_id] * pad_len
            
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# ── Training Loop ─────────────────────────────────────────────────────────────
def main():
    print("========================================")
    print("NudiLLM v1 - Instruction Fine-Tuning")
    print("========================================")

    # 1. Load Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER_PATH)
    print("✅ Tokenizer loaded.")

    # 2. Load Dataset
    dataset = InstructDataset(INSTRUCT_DATA_PATH, sp, CONFIG["context_length"])
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    print(f"✅ Dataset loaded. {len(dataset)} instruction pairs.")

    # 3. Load v0 Base Model
    v0_path = CHECKPOINT_DIR / "nudi_v0_best.pt"
    if not v0_path.exists():
        print(f"❌ Could not find {v0_path}. Did you rename the base model?")
        sys.exit(1)

    print(f"Loading base model from {v0_path}...")
    ckpt = torch.load(v0_path, map_location=CONFIG["device"])
    
    model_config = NudiConfig(**ckpt["config"])
    model = NudiLLM(model_config).to(CONFIG["device"])
    model.load_state_dict(ckpt["model"])
    print("✅ Model loaded.")

    # 4. Setup Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    # 5. Train
    model.train()
    print("\n🚀 Starting Fine-Tuning...")
    
    for epoch in range(CONFIG["epochs"]):
        epoch_loss = 0.0
        
        for x, y in dataloader:
            x, y = x.to(CONFIG["device"]), y.to(CONFIG["device"])
            
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            
            # Clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {avg_loss:.4f}")

    # 6. Save v1 Model
    v1_path = CHECKPOINT_DIR / "best.pt" # Overwrite default best.pt so UI uses it automatically
    torch.save({
        "model": model.state_dict(),
        "config": model_config.__dict__,
    }, v1_path)
    
    print("\n========================================")
    print("🎉 Fine-Tuning Complete!")
    print(f"Model saved to {v1_path}")
    print("NudiLLM v1 is ready to chat.")

if __name__ == "__main__":
    main()
