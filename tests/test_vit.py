"""Tests for patch embedding, the Vision Transformer and class token attention."""

import pytest
import torch
from torch import nn

from dlv.vit import (
    PatchEmbedding,
    VisionTransformer,
    build_pretrained_vit,
    class_token_attention,
    fine_tuning_groups,
)


def shuffle_patches(images, patch_size, permutation):
    """Reorders the non-overlapping patches of a batch of images by a permutation."""
    batch, channels, height, width = images.shape
    grid = height // patch_size
    patches = images.reshape(batch, channels, grid, patch_size, grid, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(batch, grid * grid, channels, patch_size, patch_size)
    patches = patches[:, permutation]
    patches = patches.reshape(batch, grid, grid, channels, patch_size, patch_size)
    return patches.permute(0, 3, 1, 4, 2, 5).reshape(batch, channels, height, width)


def small_vit(**overrides):
    """Builds a small Vision Transformer in evaluation mode for fast tests."""
    settings = dict(depth=2, embed_dim=24, num_heads=3, mlp_dim=48, dropout=0.0)
    settings.update(overrides)
    model = VisionTransformer(**settings)
    model.eval()
    return model


def test_patch_embedding_shape():
    """A 64x64 image with patch 8 becomes 64 tokens of size 192."""
    embedding = PatchEmbedding(image_size=64, patch_size=8, in_channels=3, embed_dim=192)
    assert embedding(torch.randn(2, 3, 64, 64)).shape == (2, 64, 192)


def test_patch_embedding_rejects_wrong_size_naming_both():
    """A wrong input size raises ValueError that mentions the expected and received sizes."""
    embedding = PatchEmbedding(image_size=64, patch_size=8, in_channels=3, embed_dim=16)
    with pytest.raises(ValueError, match=r"64.*32|32.*64"):
        embedding(torch.randn(1, 3, 32, 32))


def test_patch_embedding_matches_a_linear_projection_of_each_patch():
    """Each token is a linear function of its own flattened patch only."""
    torch.manual_seed(0)
    embedding = PatchEmbedding(image_size=16, patch_size=8, in_channels=3, embed_dim=5)
    image = torch.randn(1, 3, 16, 16)
    tokens = embedding(image)
    changed = image.clone()
    changed[:, :, 8:, 8:] += 1.0
    other = embedding(changed)
    assert torch.allclose(tokens[:, :3], other[:, :3], atol=1e-6)
    assert not torch.allclose(tokens[:, 3], other[:, 3])


def test_vit_logits_shape():
    """The default model returns one logit per class."""
    model = VisionTransformer()
    model.eval()
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 8)


def test_class_token_and_positional_embedding_are_trainable_parameters():
    """Both are registered parameters that require gradients."""
    model = small_vit()
    names = dict(model.named_parameters())
    assert names["class_token"].requires_grad
    assert names["positional_embedding"].requires_grad
    assert model.positional_embedding.shape == (1, 65, 24)


def test_positional_embedding_can_be_disabled():
    """Without positional embedding the parameter does not exist."""
    model = small_vit(use_positional_embedding=False)
    assert "positional_embedding" not in dict(model.named_parameters())


def test_return_attention_gives_one_map_per_block():
    """The attention list has depth entries shaped (batch, heads, tokens, tokens)."""
    model = VisionTransformer(depth=2, dropout=0.0)
    model.eval()
    logits, attention = model(torch.randn(2, 3, 64, 64), return_attention=True)
    assert logits.shape == (2, 8)
    assert len(attention) == 2
    assert all(weights.shape == (2, 3, 65, 65) for weights in attention)


def test_without_positional_embedding_patch_order_does_not_matter():
    """Shuffling the patches leaves the logits unchanged when no position is encoded."""
    torch.manual_seed(0)
    model = small_vit(use_positional_embedding=False)
    images = torch.randn(2, 3, 64, 64)
    permutation = torch.randperm(64)
    with torch.no_grad():
        original = model(images)
        shuffled = model(shuffle_patches(images, 8, permutation))
    assert torch.allclose(original, shuffled, atol=1e-5)


def test_with_positional_embedding_patch_order_matters():
    """Shuffling the patches changes the logits when positions are encoded."""
    torch.manual_seed(0)
    model = small_vit(use_positional_embedding=True)
    images = torch.randn(2, 3, 64, 64)
    permutation = torch.randperm(64)
    with torch.no_grad():
        original = model(images)
        shuffled = model(shuffle_patches(images, 8, permutation))
    assert (original - shuffled).abs().max() > 1e-4


def test_class_token_attention_reads_the_class_token_row():
    """The map is the class token's row of the weights, not the column other tokens aim at it."""
    attention = torch.arange(2 * 3 * 65 * 65.0).reshape(2, 3, 65, 65)
    maps = class_token_attention(attention, 8)
    assert torch.equal(maps.flatten(2), attention[:, :, 0, 1:])


def test_class_token_attention_shape_and_mass():
    """The class token map is (batch, heads, grid, grid) and sums to at most one per head."""
    model = small_vit()
    _, attention = model(torch.randn(2, 3, 64, 64), return_attention=True)
    maps = class_token_attention(attention[-1], grid_size=8)
    assert maps.shape == (2, 3, 8, 8)
    assert (maps.flatten(2).sum(dim=-1) <= 1.0 + 1e-6).all()
    assert (maps >= 0).all()


def test_gradients_reach_the_class_token_and_head():
    """A backward pass produces gradients on the class token and the classifier."""
    model = small_vit()
    model(torch.randn(2, 3, 64, 64)).sum().backward()
    assert model.class_token.grad is not None
    assert model.head.weight.grad is not None


def test_pretrained_vit_gets_a_new_head_with_one_output_per_class():
    """The classification head is replaced and every layer stays trainable."""
    model = build_pretrained_vit(num_classes=8, weights=None)
    assert isinstance(model.heads.head, nn.Linear)
    assert model.heads.head.out_features == 8
    assert all(parameter.requires_grad for parameter in model.parameters())
    model.eval()
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 224, 224)).shape == (1, 8)


def test_fine_tuning_groups_split_head_from_backbone_without_overlap():
    """The head gets its own learning rate and every parameter is in exactly one group."""
    model = build_pretrained_vit(num_classes=8, weights=None)
    backbone, head = fine_tuning_groups(model, backbone_lr=3e-5, head_lr=1e-3)
    assert backbone["lr"] == 3e-5
    assert head["lr"] == 1e-3
    head_ids = {id(parameter) for parameter in head["params"]}
    backbone_ids = {id(parameter) for parameter in backbone["params"]}
    assert head_ids == {id(parameter) for parameter in model.heads.parameters()}
    assert not head_ids & backbone_ids
    assert len(head_ids | backbone_ids) == len(list(model.parameters()))
    optimizer = torch.optim.AdamW([backbone, head], weight_decay=0.05)
    assert [group["lr"] for group in optimizer.param_groups] == [3e-5, 1e-3]
