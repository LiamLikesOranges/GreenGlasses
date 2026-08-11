"""
OrangeGlasses / GreenGlasses GPT model.

A small GPT-style decoder-only Transformer implemented
from scratch using PyTorch building blocks.

Architecture:

tokens
    ↓
token embedding + position embedding
    ↓
Transformer blocks
    ↓
final LayerNorm
    ↓
language-model head
    ↓
next-token logits
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================
# CAUSAL SELF ATTENTION
# ============================================================

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Each token can attend to itself and previous tokens,
    but never future tokens.
    """

    def __init__(
        self,
        n_embed,
        n_head,
        block_size,
        dropout,
    ):
        super().__init__()

        if n_embed % n_head != 0:
            raise ValueError(
                "n_embed must be divisible by n_head"
            )

        self.n_head = n_head
        self.head_size = n_embed // n_head

        # Query + Key + Value projection
        self.qkv_proj = nn.Linear(
            n_embed,
            3 * n_embed,
        )

        # Output projection
        self.out_proj = nn.Linear(
            n_embed,
            n_embed,
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal attention mask.
        #
        # Position i can only see positions <= i.
        mask = torch.tril(
            torch.ones(
                block_size,
                block_size,
            )
        )

        self.register_buffer(
            "mask",
            mask.view(
                1,
                1,
                block_size,
                block_size,
            ),
        )

    def forward(self, x):

        B, T, C = x.shape

        # ----------------------------------------------------
        # QKV
        # ----------------------------------------------------

        qkv = self.qkv_proj(x)

        q, k, v = qkv.split(
            C,
            dim=2,
        )

        # ----------------------------------------------------
        # Split into attention heads
        # ----------------------------------------------------

        q = q.view(
            B,
            T,
            self.n_head,
            self.head_size,
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.n_head,
            self.head_size,
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.n_head,
            self.head_size,
        ).transpose(1, 2)

        # ----------------------------------------------------
        # Attention scores
        # ----------------------------------------------------

        att = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(
            self.head_size
        )

        # Causal mask
        att = att.masked_fill(
            self.mask[
                :, :, :T, :T
            ] == 0,
            float("-inf"),
        )

        # Softmax
        att = F.softmax(
            att,
            dim=-1,
        )

        att = self.attn_dropout(att)

        # ----------------------------------------------------
        # Weighted values
        # ----------------------------------------------------

        out = att @ v

        # Recombine heads
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        # Output projection
        out = self.out_proj(out)

        out = self.resid_dropout(out)

        return out


# ============================================================
# FEED FORWARD
# ============================================================

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """

    def __init__(
        self,
        n_embed,
        dropout,
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                n_embed,
                4 * n_embed,
            ),

            nn.GELU(),

            nn.Linear(
                4 * n_embed,
                n_embed,
            ),

            nn.Dropout(
                dropout
            ),
        )

    def forward(self, x):

        return self.net(x)


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):
    """
    One Transformer decoder block.

    LayerNorm
        ↓
    Attention
        ↓
    Residual

    LayerNorm
        ↓
    Feed Forward
        ↓
    Residual
    """

    def __init__(
        self,
        n_embed,
        n_head,
        block_size,
        dropout,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(
            n_embed
        )

        self.attn = CausalSelfAttention(
            n_embed,
            n_head,
            block_size,
            dropout,
        )

        self.ln2 = nn.LayerNorm(
            n_embed
        )

        self.ff = FeedForward(
            n_embed,
            dropout,
        )

    def forward(self, x):

        # Attention residual
        x = x + self.attn(
            self.ln1(x)
        )

        # Feed-forward residual
        x = x + self.ff(
            self.ln2(x)
        )

        return x


# ============================================================
# GREENGLASSES / ORANGEGLASSES GPT
# ============================================================

class GreenGlassesGPT(nn.Module):
    """
    Decoder-only GPT-style language model.

    Despite the historical class name, this is also used
    by OrangeGlasses.
    """

    def __init__(
        self,
        vocab_size,
        n_embed,
        n_head,
        n_layer,
        block_size,
        dropout,
    ):
        super().__init__()

        self.block_size = block_size
        self.vocab_size = vocab_size

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            n_embed,
        )

        self.position_embedding = nn.Embedding(
            block_size,
            n_embed,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        # ----------------------------------------------------
        # Transformer blocks
        # ----------------------------------------------------

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    n_embed,
                    n_head,
                    block_size,
                    dropout,
                )
                for _ in range(n_layer)
            ]
        )

        # ----------------------------------------------------
        # Final layers
        # ----------------------------------------------------

        self.ln_final = nn.LayerNorm(
            n_embed
        )

        self.lm_head = nn.Linear(
            n_embed,
            vocab_size,
            bias=False,
        )

        # ----------------------------------------------------
        # Initialization
        # ----------------------------------------------------

        self.apply(
            self._init_weights
        )

    # ========================================================
    # WEIGHT INITIALIZATION
    # ========================================================

    def _init_weights(
        self,
        module,
    ):

        if isinstance(
            module,
            nn.Linear,
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding,
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        idx,
        targets=None,
    ):
        """
        Forward pass.

        idx:
            (batch, sequence_length)

        targets:
            Optional next-token targets.

        Returns:
            logits, loss
        """

        B, T = idx.shape

        if T > self.block_size:

            raise ValueError(
                f"Sequence length {T} exceeds "
                f"block_size {self.block_size}"
            )

        # ----------------------------------------------------
        # Token embeddings
        # ----------------------------------------------------

        tok_emb = self.token_embedding(
            idx
        )

        # ----------------------------------------------------
        # Position embeddings
        # ----------------------------------------------------

        positions = torch.arange(
            T,
            device=idx.device,
        )

        pos_emb = self.position_embedding(
            positions
        )

        # ----------------------------------------------------
        # Combine embeddings
        # ----------------------------------------------------

        x = tok_emb + pos_emb

        x = self.dropout(x)

        # ----------------------------------------------------
        # Transformer
        # ----------------------------------------------------

        x = self.blocks(x)

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        x = self.ln_final(x)

        # ----------------------------------------------------
        # Language model head
        # ----------------------------------------------------

        logits = self.lm_head(x)

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                targets.reshape(-1),
            )

        return logits, loss

    # ========================================================
    # GENERATION
    # ========================================================

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=0.8,
        top_k=None,
        eos_token_id=None,
    ):
        """
        Generate new tokens autoregressively.

        Parameters
        ----------
        idx:
            Starting token IDs with shape
            (batch, sequence_length).

        max_new_tokens:
            Maximum number of tokens to generate.

        temperature:
            Controls randomness.

        top_k:
            If provided, only the top K tokens
            are considered at each step.

        eos_token_id:
            Optional token ID that stops generation
            when produced.

        Returns
        -------
        Tensor containing the original prompt plus
        generated tokens.
        """

        # Make sure we're generating in evaluation mode.
        was_training = self.training

        self.eval()

        try:

            for _ in range(
                max_new_tokens
            ):

                # ------------------------------------------------
                # Keep only the model's context window.
                # ------------------------------------------------

                idx_cond = idx[
                    :,
                    -self.block_size:,
                ]

                # ------------------------------------------------
                # Forward pass
                # ------------------------------------------------

                logits, _ = self(
                    idx_cond
                )

                # We only care about predictions
                # for the final token.
                logits = logits[
                    :,
                    -1,
                    :,
                ]

                # ------------------------------------------------
                # Temperature
                # ------------------------------------------------

                temperature = max(
                    float(temperature),
                    1e-5,
                )

                logits = (
                    logits
                    / temperature
                )

                # ------------------------------------------------
                # Top-K sampling
                # ------------------------------------------------

                if top_k is not None:

                    top_k = min(
                        int(top_k),
                        logits.size(-1),
                    )

                    values, _ = torch.topk(
                        logits,
                        top_k,
                    )

                    minimum_value = values[
                        :,
                        [-1],
                    ]

                    logits = torch.where(
                        logits
                        < minimum_value,
                        torch.full_like(
                            logits,
                            float("-inf"),
                        ),
                        logits,
                    )

                # ------------------------------------------------
                # Convert logits -> probabilities
                # ------------------------------------------------

                probs = F.softmax(
                    logits,
                    dim=-1,
                )

                # ------------------------------------------------
                # Sample next token
                # ------------------------------------------------

                next_token = torch.multinomial(
                    probs,
                    num_samples=1,
                )

                # ------------------------------------------------
                # Append token
                # ------------------------------------------------

                idx = torch.cat(
                    (
                        idx,
                        next_token,
                    ),
                    dim=1,
                )

                # ------------------------------------------------
                # EOS stopping
                # ------------------------------------------------

                if (
                    eos_token_id is not None
                    and torch.all(
                        next_token
                        == eos_token_id
                    )
                ):

                    break

        finally:

            # Restore the previous training state.
            if was_training:

                self.train()

        return idx

    # ========================================================
    # PARAMETER COUNT
    # ========================================================

    def num_params(
        self,
    ):

        return sum(
            p.numel()
            for p in self.parameters()
        )


# ============================================================
# OPTIONAL ALIAS
# ============================================================

# This lets future code refer to the model as OrangeGlassesGPT
# without breaking older GreenGlasses code.

OrangeGlassesGPT = GreenGlassesGPT