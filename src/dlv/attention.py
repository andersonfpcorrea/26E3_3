"""Attention mechanism and Transformer encoder block built from linear layers and softmax."""

from math import sqrt

import torch
from torch import Tensor, nn


def scaled_dot_product_attention(
    query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Computes softmax(Q K^T / sqrt(d)) V and returns the output with the weights.

    Args:
        query: tensor shaped (..., queries, d).
        key: tensor shaped (..., keys, d).
        value: tensor shaped (..., keys, d_value).
        mask: optional boolean tensor broadcastable to (..., queries, keys),
            true where attention is allowed.

    A query with no allowed key gets all-zero weights and a zero output
    instead of NaN, and its gradients stay finite.
    """
    scores = query @ key.transpose(-2, -1) / sqrt(query.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    if mask is not None:
        weights = weights * mask
    return weights @ value, weights


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention where every head owns its query, key and value layers.

    Each head projects the input from embed_dim to embed_dim // num_heads with
    separate linear layers. Head outputs are concatenated and mixed by one
    output projection.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        """Creates the per-head projections and the output projection."""
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
            )
        head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.query = nn.ModuleList(nn.Linear(embed_dim, head_dim) for _ in range(num_heads))
        self.key = nn.ModuleList(nn.Linear(embed_dim, head_dim) for _ in range(num_heads))
        self.value = nn.ModuleList(nn.Linear(embed_dim, head_dim) for _ in range(num_heads))
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, return_weights: bool = False):
        """Attends over the tokens of x shaped (batch, tokens, embed_dim).

        Returns the output, or (output, weights) with weights shaped
        (batch, heads, tokens, tokens) when return_weights is true.
        """
        outputs, all_weights = [], []
        for query, key, value in zip(self.query, self.key, self.value):
            output, weights = scaled_dot_product_attention(query(x), key(x), value(x))
            outputs.append(output)
            all_weights.append(weights)
        merged = self.output_projection(torch.cat(outputs, dim=-1))
        merged = self.dropout(merged)
        if return_weights:
            return merged, torch.stack(all_weights, dim=1)
        return merged


class TransformerEncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block: attention and feedforward, each with a residual."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.0):
        """Creates the two layer norms, the attention and the two-layer feedforward."""
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feedforward_norm = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, return_weights: bool = False):
        """Applies the block to x shaped (batch, tokens, embed_dim).

        Returns the output, or (output, weights) when return_weights is true.
        """
        attended, weights = self.attention(self.attention_norm(x), return_weights=True)
        x = x + attended
        x = x + self.feedforward(self.feedforward_norm(x))
        if return_weights:
            return x, weights
        return x
