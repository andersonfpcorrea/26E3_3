"""Class-conditional GAN for 64x64 grayscale images, its losses, training loop and two diagnostic measures."""

from contextlib import contextmanager

import torch
from torch import Tensor, nn
from torch.nn.functional import binary_cross_entropy_with_logits, relu
from torch.nn.utils import parametrizations
from torchvision.utils import make_grid

IMAGE_SIZE = 64
LOSS_KINDS = ("bce", "hinge")
DIVERSITY_SAMPLES = 32
DEFAULT_SNAPSHOT_PER_CLASS = 8
SNAPSHOT_COLUMNS = 8


def check_labels(labels: Tensor, num_classes: int) -> None:
    """Raises ValueError when any label falls outside 0 to num_classes - 1.

    Reads the label range back to Python, which synchronises with the GPU.
    """
    if labels.numel() == 0:
        return
    lowest, highest = int(labels.min()), int(labels.max())
    if lowest < 0 or highest >= num_classes:
        raise ValueError(
            f"labels must be in [0, {num_classes - 1}], got values from {lowest} to {highest}"
        )


def initialise_weights(module: nn.Module) -> None:
    """Draws convolution and batch normalisation weights from N(0, 0.02), the DCGAN recipe.

    Layers already wrapped in a parametrization are skipped; they are initialised before wrapping.
    """
    if hasattr(module, "parametrizations"):
        return
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


class ConditionalGenerator(nn.Module):
    """Maps noise and a class label to a 64x64 grayscale image in the range -1 to 1.

    The label is embedded to the size of the noise and concatenated with it, so
    the transposed convolutions see which class to draw from the first layer on.
    """

    def __init__(self, latent_dim: int = 100, num_classes: int = 3, base_channels: int = 64):
        """Builds the label embedding and four doublings from 4x4 to 64x64."""
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.validate_labels = True
        self.embedding = nn.Embedding(num_classes, latent_dim)
        width = base_channels
        self.network = nn.Sequential(
            nn.ConvTranspose2d(2 * latent_dim, width * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(width * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(width * 8, width * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(width * 4, width * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(width * 2, width, 4, 2, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(True),
            nn.ConvTranspose2d(width, 1, 4, 2, 1),
            nn.Tanh(),
        )
        self.network.apply(initialise_weights)

    def forward(self, noise: Tensor, labels: Tensor) -> Tensor:
        """Maps (n, latent_dim) noise and (n,) integer labels to (n, 1, 64, 64) images."""
        if self.validate_labels:
            check_labels(labels, self.num_classes)
        conditioned = torch.cat([noise, self.embedding(labels)], dim=1)
        return self.network(conditioned[:, :, None, None])


class ConditionalDiscriminator(nn.Module):
    """Scores a 64x64 grayscale image given its class label with one logit.

    The label becomes a learned 64x64 map stacked on the image as a second
    channel. With spectral normalisation every weight layer has its spectral norm
    bounded and batch normalisation is left out, since the two interfere.
    """

    def __init__(
        self, num_classes: int = 3, base_channels: int = 64, spectral_norm: bool = False
    ):
        """Builds the label map and the strided convolutions from 64x64 down to one logit."""
        super().__init__()
        self.num_classes = num_classes
        self.validate_labels = True
        self.embedding = nn.Embedding(num_classes, IMAGE_SIZE * IMAGE_SIZE)
        width = base_channels
        sizes = [2, width, width * 2, width * 4, width * 8]
        layers = []
        for index in range(4):
            layers.append(
                self.weight_layer(
                    nn.Conv2d(sizes[index], sizes[index + 1], 4, 2, 1, bias=spectral_norm),
                    spectral_norm,
                )
            )
            if not spectral_norm and index > 0:
                layers.append(nn.BatchNorm2d(sizes[index + 1]))
            layers.append(nn.LeakyReLU(0.2, True))
        layers.append(self.weight_layer(nn.Conv2d(sizes[4], 1, 4, 1, 0), spectral_norm))
        self.network = nn.Sequential(*layers)
        self.network.apply(initialise_weights)

    @staticmethod
    def weight_layer(layer: nn.Module, spectral_norm: bool) -> nn.Module:
        """Initialises a convolution and wraps it in spectral normalisation when asked."""
        initialise_weights(layer)
        return parametrizations.spectral_norm(layer) if spectral_norm else layer

    def forward(self, images: Tensor, labels: Tensor) -> Tensor:
        """Maps (n, 1, 64, 64) images and (n,) labels to (n,) logits."""
        if self.validate_labels:
            check_labels(labels, self.num_classes)
        label_map = self.embedding(labels).view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        return self.network(torch.cat([images, label_map], dim=1)).flatten()


def discriminator_loss(real_logits: Tensor, fake_logits: Tensor, kind: str) -> Tensor:
    """Loss the discriminator minimises: binary cross-entropy or hinge, summed over real and fake."""
    if kind == "bce":
        return binary_cross_entropy_with_logits(
            real_logits, torch.ones_like(real_logits)
        ) + binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    if kind == "hinge":
        return relu(1.0 - real_logits).mean() + relu(1.0 + fake_logits).mean()
    raise ValueError(f"loss kind must be one of {LOSS_KINDS}, got {kind!r}")


def generator_loss(fake_logits: Tensor, kind: str) -> Tensor:
    """Loss the generator minimises: non-saturating binary cross-entropy or the hinge generator term."""
    if kind == "bce":
        return binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    if kind == "hinge":
        return -fake_logits.mean()
    raise ValueError(f"loss kind must be one of {LOSS_KINDS}, got {kind!r}")


def unit_range(images: Tensor) -> Tensor:
    """Returns images as floats in 0 to 1: uint8 is divided by 255, floats are taken as already scaled."""
    if images.dtype == torch.uint8:
        return images.float() / 255.0
    return images.float()


def to_uint8(images: Tensor) -> Tensor:
    """Converts generator output in -1 to 1 into uint8 images in 0 to 255."""
    return ((images + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)


def pairwise_diversity(images: Tensor) -> float:
    """Mean absolute pixel difference between all pairs of images, in 0 to 1.

    Zero when every image is identical, so a value near zero on generated
    images signals mode collapse. Accepts uint8 or floats already in 0 to 1.
    """
    count = images.shape[0]
    if count < 2:
        return 0.0
    flat = unit_range(images).reshape(count, -1)
    distances = torch.cdist(flat, flat, p=1.0) / flat.shape[1]
    return float(distances.sum() / (count * (count - 1)))


def nearest_real_distance(synthetic: Tensor, real: Tensor) -> Tensor:
    """Mean absolute pixel difference from each synthetic image to its closest real image, in 0 to 1.

    Zero for a synthetic image that is a real one, so small values across a
    whole batch point at memorisation of the training set.
    """
    synthetic_flat = unit_range(synthetic).reshape(synthetic.shape[0], -1)
    real_flat = unit_range(real).reshape(real.shape[0], -1)
    distances = torch.cdist(synthetic_flat, real_flat, p=1.0) / synthetic_flat.shape[1]
    return distances.min(dim=1).values


def image_grid(images: Tensor) -> Tensor:
    """Tiles uint8 (n, 1, h, w) images into one uint8 (1, H, W) grid."""
    return make_grid(images, nrow=SNAPSHOT_COLUMNS, padding=2)[:1]


def sample_images(generator: ConditionalGenerator, noise: Tensor, labels: Tensor) -> Tensor:
    """Generates uint8 images with the generator in evaluation mode, restoring its previous mode."""
    was_training = generator.training
    generator.eval()
    with torch.no_grad():
        images = to_uint8(generator(noise, labels))
    generator.train(was_training)
    return images


@contextmanager
def unchecked_labels(*models: nn.Module):
    """Turns the per-forward label range check off for the models, restoring it on exit."""
    for model in models:
        model.validate_labels = False
    try:
        yield
    finally:
        for model in models:
            model.validate_labels = True


def default_snapshot_inputs(
    latent_dim: int, num_classes: int, noise_generator: torch.Generator
) -> tuple[Tensor, Tensor]:
    """Builds a fixed noise batch with the same number of images for every class."""
    labels = torch.arange(num_classes).repeat_interleave(DEFAULT_SNAPSHOT_PER_CLASS)
    noise = torch.randn(labels.shape[0], latent_dim, generator=noise_generator)
    return noise, labels


def train_gan(
    generator: ConditionalGenerator,
    discriminator: ConditionalDiscriminator,
    images: Tensor,
    labels: Tensor,
    epochs: int,
    batch_size: int,
    device,
    loss: str = "bce",
    generator_lr: float = 2e-4,
    discriminator_lr: float = 2e-4,
    betas=(0.5, 0.999),
    seed: int = 42,
    snapshot_epochs=(),
    snapshot_noise: Tensor | None = None,
    snapshot_labels: Tensor | None = None,
) -> dict:
    """Trains the two networks adversarially, one discriminator and one generator step per batch.

    The discriminator step sees real images and generated images that carry no
    gradient to the generator. The generator step draws fresh noise and scores
    its images through the updated discriminator. Each optimiser owns only its
    own network, and the discriminator computes no parameter gradients during
    the generator step. Label range checks are skipped inside the loop (the
    labels are validated once up front) and restored afterwards. Both networks train in training mode; the generator switches to
    evaluation mode for per-epoch diversity and snapshots and returns to
    training mode. `images` are uint8 (n, 1, 64, 64) and are scaled to -1..1
    here. Without snapshot noise and labels, a fixed batch with the same number
    of images per class is used.

    Returns a history with per-epoch lists `discriminator_loss`,
    `generator_loss`, `real_score` and `fake_score` (mean sigmoid of the logits),
    `diversity` (class to a per-epoch list, from fixed noise) and `snapshots`
    (epoch from 1 to uint8 grid tensors on the CPU). Drop `snapshots` before
    saving the history as JSON.

    `seed` governs the shuffling order, the training noise and the fixed
    diversity and snapshot noise, all drawn from local generators; the global
    random state is left untouched. Initialising the networks is the caller's
    job and is not covered by `seed`.
    """
    if loss not in LOSS_KINDS:
        raise ValueError(f"loss kind must be one of {LOSS_KINDS}, got {loss!r}")
    if (snapshot_noise is None) != (snapshot_labels is None):
        raise ValueError("snapshot_noise and snapshot_labels must be given together")
    check_labels(labels, generator.num_classes)
    shuffle_generator = torch.Generator().manual_seed(seed)
    fixed_generator = torch.Generator().manual_seed(seed + 1)
    noise_generator = torch.Generator(device=device).manual_seed(seed)
    num_classes = generator.num_classes
    latent_dim = generator.latent_dim

    generator.to(device)
    discriminator.to(device)
    real_images = images.to(device)
    real_labels = labels.to(device).long()
    count = real_images.shape[0]

    diversity_labels = torch.arange(num_classes).repeat_interleave(DIVERSITY_SAMPLES)
    diversity_noise = torch.randn(diversity_labels.shape[0], latent_dim, generator=fixed_generator)
    diversity_noise, diversity_labels = diversity_noise.to(device), diversity_labels.to(device)
    if snapshot_noise is None:
        snapshot_noise, snapshot_labels = default_snapshot_inputs(
            latent_dim, num_classes, fixed_generator
        )
    snapshot_noise, snapshot_labels = snapshot_noise.to(device), snapshot_labels.to(device).long()
    snapshot_set = set(snapshot_epochs)

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=generator_lr, betas=betas)
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=discriminator_lr, betas=betas
    )

    history = {
        "discriminator_loss": [],
        "generator_loss": [],
        "real_score": [],
        "fake_score": [],
        "diversity": {index: [] for index in range(num_classes)},
        "snapshots": {},
    }
    with unchecked_labels(generator, discriminator):
        generator.train()
        discriminator.train()

        for epoch in range(1, epochs + 1):
            order = torch.randperm(count, generator=shuffle_generator).to(device)
            totals = torch.zeros(4, device=device)
            for start in range(0, count, batch_size):
                indices = order[start : start + batch_size]
                batch = indices.shape[0]
                real = real_images[indices].float() / 127.5 - 1.0
                batch_labels = real_labels[indices]

                noise = torch.randn(batch, latent_dim, device=device, generator=noise_generator)
                with torch.no_grad():
                    fake = generator(noise, batch_labels)
                real_logits = discriminator(real, batch_labels)
                fake_logits = discriminator(fake, batch_labels)
                discriminator_step_loss = discriminator_loss(real_logits, fake_logits, loss)
                discriminator_optimizer.zero_grad(set_to_none=True)
                discriminator_step_loss.backward()
                discriminator_optimizer.step()

                noise = torch.randn(batch, latent_dim, device=device, generator=noise_generator)
                discriminator.requires_grad_(False)
                generated_logits = discriminator(generator(noise, batch_labels), batch_labels)
                generator_step_loss = generator_loss(generated_logits, loss)
                generator_optimizer.zero_grad(set_to_none=True)
                generator_step_loss.backward()
                generator_optimizer.step()
                discriminator.requires_grad_(True)

                with torch.no_grad():
                    totals += batch * torch.stack(
                        [
                            discriminator_step_loss.detach(),
                            generator_step_loss.detach(),
                            torch.sigmoid(real_logits).mean(),
                            torch.sigmoid(fake_logits).mean(),
                        ]
                    )

            averages = (totals / count).tolist()
            history["discriminator_loss"].append(averages[0])
            history["generator_loss"].append(averages[1])
            history["real_score"].append(averages[2])
            history["fake_score"].append(averages[3])

            diverse = sample_images(generator, diversity_noise, diversity_labels)
            for index in range(num_classes):
                members = diverse[diversity_labels == index]
                history["diversity"][index].append(pairwise_diversity(members))
            if epoch in snapshot_set:
                snapshot = sample_images(generator, snapshot_noise, snapshot_labels)
                history["snapshots"][epoch] = image_grid(snapshot).cpu()

    return history


def generate(generator: ConditionalGenerator, labels: Tensor, device, seed: int) -> Tensor:
    """Generates one uint8 (n, 1, 64, 64) image per label, deterministically for a given seed.

    Raises ValueError for a label the generator was not built for.
    """
    labels = labels.to(device).long()
    check_labels(labels, generator.num_classes)
    generator.to(device)
    noise_generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(labels.shape[0], generator.latent_dim, generator=noise_generator)
    return sample_images(generator, noise.to(device), labels).cpu()
