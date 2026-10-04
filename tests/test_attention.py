"""Tests for scaled dot-product attention, multi-head attention and the encoder block."""

import pytest
import torch
from torch import nn

from dlv.attention import (
    MultiHeadAttention,
    TransformerEncoderBlock,
    scaled_dot_product_attention,
)


def test_attention_weights_are_a_distribution_per_row():
    """Weights are non-negative, rows sum to one and the output has the value shape."""
    torch.manual_seed(0)
    query, key, value = torch.randn(2, 5, 4), torch.randn(2, 7, 4), torch.randn(2, 7, 6)
    output, weights = scaled_dot_product_attention(query, key, value)
    assert output.shape == (2, 5, 6)
    assert weights.shape == (2, 5, 7)
    assert (weights >= 0).all()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5), atol=1e-6)


def test_attention_matches_the_formula():
    """The output equals softmax(QK^T / sqrt(d)) V computed by hand."""
    torch.manual_seed(0)
    query, key, value = torch.randn(1, 3, 4), torch.randn(1, 3, 4), torch.randn(1, 3, 4)
    expected_weights = torch.softmax(query @ key.transpose(-2, -1) / 2.0, dim=-1)
    output, weights = scaled_dot_product_attention(query, key, value)
    assert torch.allclose(weights, expected_weights, atol=1e-6)
    assert torch.allclose(output, expected_weights @ value, atol=1e-6)


def test_masked_position_gets_zero_weight():
    """A position where the mask is false receives exactly zero weight."""
    torch.manual_seed(0)
    query, key, value = torch.randn(1, 3, 4), torch.randn(1, 3, 4), torch.randn(1, 3, 4)
    mask = torch.tensor([[[True, True, False]]]).expand(1, 3, 3)
    _, weights = scaled_dot_product_attention(query, key, value, mask)
    assert (weights[..., 2] == 0).all()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 3), atol=1e-6)


def test_fully_masked_row_gives_zeros_without_nan():
    """A row with nothing allowed yields zero weights and zero output, never NaN."""
    torch.manual_seed(0)
    query, key, value = torch.randn(1, 2, 4), torch.randn(1, 3, 4), torch.randn(1, 3, 4)
    mask = torch.tensor([[[True, True, True], [False, False, False]]])
    output, weights = scaled_dot_product_attention(query, key, value, mask)
    assert not torch.isnan(output).any()
    assert not torch.isnan(weights).any()
    assert (weights[0, 1] == 0).all()
    assert (output[0, 1] == 0).all()


def test_fully_masked_row_keeps_gradients_finite():
    """Backpropagating through a fully masked row produces no NaN gradient."""
    query = torch.randn(1, 2, 4, requires_grad=True)
    key, value = torch.randn(1, 3, 4), torch.randn(1, 3, 4)
    mask = torch.tensor([[[True, True, True], [False, False, False]]])
    output, _ = scaled_dot_product_attention(query, key, value, mask)
    output.sum().backward()
    assert torch.isfinite(query.grad).all()


def test_identical_keys_give_uniform_weights():
    """When every key is the same, attention is spread evenly."""
    query, value = torch.randn(1, 3, 4), torch.randn(1, 5, 4)
    key = torch.ones(1, 5, 4)
    _, weights = scaled_dot_product_attention(query, key, value)
    assert torch.allclose(weights, torch.full((1, 3, 5), 0.2), atol=1e-6)


def test_multi_head_shapes():
    """Output keeps the input shape and weights are (batch, heads, tokens, tokens)."""
    attention = MultiHeadAttention(embed_dim=12, num_heads=3)
    x = torch.randn(2, 5, 12)
    assert attention(x).shape == (2, 5, 12)
    output, weights = attention(x, return_weights=True)
    assert output.shape == (2, 5, 12)
    assert weights.shape == (2, 3, 5, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 3, 5), atol=1e-6)


def test_multi_head_has_one_projection_per_head_and_role():
    """The module holds three linear layers per head plus one output projection."""
    attention = MultiHeadAttention(embed_dim=12, num_heads=3)
    linears = [module for module in attention.modules() if isinstance(module, nn.Linear)]
    assert len(linears) == 3 * 3 + 1
    head_layers = [layer for layer in linears if layer.out_features == 4]
    assert len(head_layers) == 9
    assert all(layer.in_features == 12 for layer in head_layers)


def test_changing_one_head_leaves_other_heads_unchanged():
    """Editing the projections of head 1 changes only head 1's weights."""
    torch.manual_seed(0)
    attention = MultiHeadAttention(embed_dim=12, num_heads=3)
    x = torch.randn(2, 5, 12)
    _, before = attention(x, return_weights=True)
    with torch.no_grad():
        for layers in (attention.query, attention.key, attention.value):
            layers[1].weight.add_(torch.randn_like(layers[1].weight))
    _, after = attention(x, return_weights=True)
    assert torch.equal(before[:, 0], after[:, 0])
    assert torch.equal(before[:, 2], after[:, 2])
    assert not torch.allclose(before[:, 1], after[:, 1])


def test_multi_head_rejects_indivisible_embed_dim():
    """An embedding size not divisible by the head count raises ValueError."""
    with pytest.raises(ValueError):
        MultiHeadAttention(embed_dim=10, num_heads=3)


def test_encoder_block_preserves_shape():
    """The block maps (batch, tokens, dim) to the same shape and returns weights on request."""
    block = TransformerEncoderBlock(embed_dim=12, num_heads=3, mlp_dim=24)
    x = torch.randn(2, 5, 12)
    assert block(x).shape == (2, 5, 12)
    output, weights = block(x, return_weights=True)
    assert output.shape == (2, 5, 12)
    assert weights.shape == (2, 3, 5, 5)


def test_encoder_block_is_identity_when_final_projections_are_zero():
    """With both sublayers' last projections zeroed, only the residual paths remain."""
    block = TransformerEncoderBlock(embed_dim=12, num_heads=3, mlp_dim=24)
    block.eval()
    with torch.no_grad():
        for layer in (block.attention.output_projection, block.feedforward[3]):
            layer.weight.zero_()
            layer.bias.zero_()
    x = torch.randn(2, 5, 12)
    assert torch.allclose(block(x), x, atol=1e-6)


def test_gradients_reach_every_parameter():
    """Every parameter of the block receives a finite gradient."""
    torch.manual_seed(0)
    block = TransformerEncoderBlock(embed_dim=12, num_heads=3, mlp_dim=24)
    block(torch.randn(2, 5, 12)).sum().backward()
    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
