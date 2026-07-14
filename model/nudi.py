"""
NudiLLM - The Mini Kannada GPT Model Architecture
A decoder-only Transformer, inspired by GPT-2/nanoGPT.

Designed for GTX 1650 (4GB VRAM):
  - ~20M parameters
  - FP16 training (no BF16 on Turing)
  - Fits comfortably in 4GB with batch_size=8, seq_len=256
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class NudiConfig:
    """
    Configuration for the mini NudiLLM model.
    Tuned for GTX 1650 4GB VRAM.
    
    Parameter count: ~20M
    VRAM usage (FP16, batch=8, seq=256): ~2.5GB
    """
    # Architecture
    vocab_size: int = 8000       # Matches our Kannada BPE tokenizer
    context_length: int = 256    # Max sequence length (tokens)
    n_embd: int = 384            # Embedding dimension
    n_heads: int = 6             # Attention heads (n_embd must be divisible)
    n_layers: int = 6            # Number of transformer blocks
    dropout: float = 0.1         # Dropout rate

    # Special tokens
    pad_id: int = 0
    bos_id: int = 2
    eos_id: int = 3


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.
    'Causal' = each token can only attend to previous tokens (left-to-right).
    This is the core of a GPT-style language model.

    Uses PyTorch Flash Attention (F.scaled_dot_product_attention) which:
      - Never materialises the full T×T attention matrix
      - Saves ~300-500MB VRAM vs manual attention
      - Handles the causal mask internally via is_causal=True
      - Works on GTX 1650 via the math fallback (PyTorch 2.0+)
    """

    def __init__(self, config: NudiConfig):
        super().__init__()
        assert config.n_embd % config.n_heads == 0, \
            "n_embd must be divisible by n_heads"

        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_heads
        self.dropout = config.dropout

        # Single linear for Q, K, V projections (3x for efficiency)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.resid_dropout = nn.Dropout(config.dropout)
        # Note: attn_dropout is passed directly to scaled_dot_product_attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # Batch, Sequence length, Channels (n_embd)

        # Compute Q, K, V all at once
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape for multi-head: (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Flash Attention: fused kernel, no T×T matrix in memory, causal mask built-in
        # dropout_p is only applied during training
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )  # (B, n_heads, T, head_dim)

        # Concatenate heads: (B, T, n_embd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        out = self.resid_dropout(self.c_proj(out))
        return out


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN).
    Two linear layers with GELU activation.
    Hidden dim = 4 * n_embd (standard GPT ratio).
    """

    def __init__(self, config: NudiConfig):
        super().__init__()
        hidden = 4 * config.n_embd
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, config.n_embd, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block:
      LayerNorm → Attention → residual
      LayerNorm → FFN → residual
    Pre-normalization (like GPT-2) for training stability.
    """

    def __init__(self, config: NudiConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm + residual connections
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class NudiLLM(nn.Module):
    """
    NudiLLM — Mini Kannada Language Model
    
    A decoder-only Transformer (GPT-2 style) trained on Kannada text.
    
    Architecture:
      Token Embeddings + Positional Embeddings
      → N × TransformerBlock
      → LayerNorm
      → Linear (logits over vocabulary)
    
    ~20M parameters, optimized for GTX 1650 (4GB VRAM).
    """

    def __init__(self, config: NudiConfig):
        super().__init__()
        self.config = config

        # Token + positional embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.context_length, config.n_embd)
        self.emb_drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final layer norm
        self.ln_final = nn.LayerNorm(config.n_embd)

        # Language model head (maps embeddings → vocabulary logits)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: token embedding and lm_head share weights
        # (standard GPT trick — reduces params and improves training)
        self.token_emb.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        # Scale residual projections (from GPT-2 paper)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor = None,
    ):
        """
        Args:
            input_ids: (B, T) token ids
            targets:   (B, T) target token ids (for training loss)
        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar CrossEntropy loss (None if targets not provided)
        """
        B, T = input_ids.shape
        assert T <= self.config.context_length, \
            f"Sequence length {T} > context_length {self.config.context_length}"

        # Token + positional embeddings
        positions = torch.arange(T, device=input_ids.device)  # (T,)
        x = self.emb_drop(
            self.token_emb(input_ids) + self.pos_emb(positions)
        )  # (B, T, n_embd)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Flatten for CrossEntropy: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=self.config.pad_id,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.3,
    ) -> torch.Tensor:
        """
        Autoregressive text generation with temperature + top-k + top-p sampling.
        
        Args:
            input_ids:          (1, T) starting token ids
            max_new_tokens:     how many new tokens to generate
            temperature:        controls randomness (lower = more deterministic)
            top_k:              keep only top-k logits
            top_p:              nucleus sampling (keep tokens summing to top_p probability)
            repetition_penalty: >1.0 penalises already-seen tokens (reduces looping)
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Truncate if context is too long
            ctx = input_ids[:, -self.config.context_length:]

            logits, _ = self(ctx)
            logits = logits[:, -1, :]  # Take last token logits: (1, vocab_size)

            # Repetition penalty — divide logits of seen tokens to make them less likely
            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            # Apply temperature
            logits = logits / temperature

            # Top-K filtering
            if top_k > 0:
                top_k_vals = torch.topk(logits, min(top_k, logits.size(-1))).values
                logits[logits < top_k_vals[:, -1:]] = float("-inf")

            # Top-P (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative prob above top_p
                sorted_idx_remove = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_idx_remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop at EOS token
            if next_token.item() == self.config.eos_id:
                break

        return input_ids

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        params = self.count_parameters()
        return (
            f"NudiLLM(\n"
            f"  vocab_size={self.config.vocab_size}, "
            f"context_length={self.config.context_length}\n"
            f"  n_embd={self.config.n_embd}, n_heads={self.config.n_heads}, "
            f"n_layers={self.config.n_layers}\n"
            f"  Parameters: {params:,} (~{params/1e6:.1f}M)\n"
            f")"
        )
