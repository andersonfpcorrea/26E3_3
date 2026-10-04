"""Vision Transformer built from patch embedding and encoder blocks."""

import torch
from torch import Tensor, nn

from dlv.attention import TransformerEncoderBlock


class PatchEmbedding(nn.Module):
    """Cuts an image into non-overlapping square patches and embeds each as a token."""

    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        """Creates the strided convolution that projects every patch linearly."""
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size {image_size} must be divisible by patch_size {patch_size}"
            )
        self.image_size = image_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, images: Tensor) -> Tensor:
        """Maps (batch, channels, size, size) to (batch, patches, embed_dim)."""
        if images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"expected images of size {self.image_size}x{self.image_size}, "
                f"got {images.shape[-2]}x{images.shape[-1]}"
            )
        return self.projection(images).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    """Vision Transformer classifier reading the class token after a stack of encoder blocks.

    Self-attention treats its input tokens as a set, so the order of the
    patches reaches the model only through the positional embedding.
    """

    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        num_classes: int = 8,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_dim: int = 768,
        dropout: float = 0.1,
        use_positional_embedding: bool = True,
    ):
        """Creates the patch embedding, class token, optional positions, blocks and head."""
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, 3, embed_dim)
        tokens = self.patch_embedding.num_patches + 1
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.class_token, std=0.02)
        if use_positional_embedding:
            self.positional_embedding = nn.Parameter(torch.zeros(1, tokens, embed_dim))
            nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        else:
            self.register_parameter("positional_embedding", None)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(
        self, images: Tensor, return_attention: bool = False
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Classifies images shaped (batch, 3, size, size).

        Returns logits, or (logits, attention) where attention lists one
        (batch, heads, tokens, tokens) tensor per block.
        """
        tokens = self.patch_embedding(images)
        class_token = self.class_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([class_token, tokens], dim=1)
        if self.positional_embedding is not None:
            tokens = tokens + self.positional_embedding
        tokens = self.dropout(tokens)
        attention = []
        for block in self.blocks:
            tokens, weights = block(tokens, return_weights=True)
            attention.append(weights)
        logits = self.head(self.norm(tokens)[:, 0])
        if return_attention:
            return logits, attention
        return logits


def class_token_attention(attention: Tensor, grid_size: int) -> Tensor:
    """Extracts the class token's attention to every patch as a spatial map.

    Args:
        attention: one block's weights shaped (batch, heads, tokens, tokens),
            with the class token at index 0.
        grid_size: patches per image side.

    Returns a tensor shaped (batch, heads, grid_size, grid_size).
    """
    patches = attention[:, :, 0, 1:]
    return patches.reshape(*patches.shape[:2], grid_size, grid_size)
