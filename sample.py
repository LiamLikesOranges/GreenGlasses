"""
A GPT-style transformer, written from scratch (only using PyTorch's basic
building blocks — nn.Linear, nn.Embedding, nn.LayerNorm — not a prebuilt
transformer module). This is the actual architecture behind every modern
LLM, just small.

Rough shape of the model for one input sequence:

  tokens -> [token embedding + position embedding] -> N x TransformerBlock
          -> final layer norm -> linear layer -> next-token probabilities

Each TransformerBlock is:

  x -> layer norm -> self-attention -> add back to x (residual)
    -> layer norm -> feed-forward   -> add back to x (residual)

Self-attention is the key idea: it lets every position in the sequence
look at every earlier position (this one is "causal" / can't peek ahead)
and decide, via learned weights, which earlier characters are relevant
to predicting the next one.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each position can only attend to
    itself and earlier positions (causal = no peeking at the future)."""

    def __init__(self, n_embed, n_head, block_size, dropout):
        super().__init__()
        assert n_embed % n_head == 0, "n_embed must be divisible by n_head"
        self.n_head = n_head
        self.head_size = n_embed // n_head

        # one linear layer produces query, key, and value all at once
        self.qkv_proj = nn.Linear(n_embed, 3 * n_embed)
        self.out_proj = nn.Linear(n_embed, n_embed)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # causal mask: position i can only attend to positions <= i
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, embedding dim

        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)

        # reshape into (B, n_head, T, head_size) so each head attends independently
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # scaled dot-product attention scores
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v  # (B, n_head, T, head_size)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # recombine heads
        out = self.resid_dropout(self.out_proj(out))
        return out


class FeedForward(nn.Module):
    """Simple two-layer MLP applied to each position independently."""

    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_embed, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ff = FeedForward(n_embed, dropout)

    def forward(self, x):
        # residual connections: add the sublayer's output back to its input.
        # this keeps gradients flowing well in deep networks.
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GreenGlassesGPT(nn.Module):
    def __init__(self, vocab_size, n_embed, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[TransformerBlock(n_embed, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_final = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"sequence length {T} exceeds block_size {self.block_size}"

        tok_emb = self.token_embedding(idx)  # (B, T, C)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T, C)
        x = self.dropout(tok_emb + pos_emb)

        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        """Autoregressively generate new tokens, one at a time, feeding each
        prediction back in as input for the next step."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # crop to context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)  # last position only

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
