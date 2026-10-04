"""Chest X-ray data preparation and per-class metrics for the COVID-19 screening case study."""

import random
from math import pi
from pathlib import Path
from statistics import fmean, stdev

import numpy as np
import torch
from PIL.Image import Resampling
from PIL.Image import open as open_image
from torch import Tensor, nn
from torch.nn.functional import affine_grid, grid_sample, interpolate
from torchvision.models import ResNet18_Weights, resnet18

from dlv.data import IMAGENET_MEAN, IMAGENET_STD, file_digest, has_image_name, is_readable_image

XRAY_CLASSES = (
    ("Normal", "Normal"),
    ("Viral Pneumonia", "Pneumonia"),
    ("COVID", "COVID-19"),
)


def find_class_root(root: Path) -> Path:
    """Locates the folder that directly holds the class folders.

    Root itself qualifies when one of the three class folders with an
    ``images`` directory sits inside it; otherwise the search goes downward
    and the shallowest such folder wins, ties broken by sorted path. Raises
    ValueError naming root and the expected layout when none is found.
    """
    root = Path(root)
    candidates = {root} if root.is_dir() else set()
    for folder, _ in XRAY_CLASSES:
        candidates.update(path.parent.parent for path in root.rglob(f"{folder}/images") if path.is_dir())
    holders = sorted(
        (path for path in candidates if any((path / folder / "images").is_dir() for folder, _ in XRAY_CLASSES)),
        key=lambda path: (len(path.relative_to(root).parts), str(path)),
    )
    if not holders:
        raise ValueError(
            f"no X-ray class folders found under {root}; expected "
            f"<dataset folder>/Normal/images, <dataset folder>/Viral Pneumonia/images "
            f"and <dataset folder>/COVID/images"
        )
    return holders[0]


def count_xray_files(root: Path) -> int:
    """Counts the files named like images directly inside ``<class folder>/images/``.

    Nothing is opened, so exact copies and unreadable files are counted too.
    This is the size of the study classes before any cleaning, with masks and
    other class folders left out.
    """
    root = find_class_root(root)
    total = 0
    for folder, _ in XRAY_CLASSES:
        images = root / folder / "images"
        if images.is_dir():
            total += sum(1 for path in images.iterdir() if path.is_file() and has_image_name(path))
    return total


def list_xray_images(root: Path) -> tuple[list[Path], list[int]]:
    """Finds the readable X-ray images of the three study classes under root.

    Root may be the folder holding the class folders or any ancestor of it;
    see ``find_class_root``.
    Only files directly inside ``<class folder>/images/`` are taken, so lung
    masks and any other class folder are never listed. The label is the index
    of the class in ``XRAY_CLASSES``. Exact copies by file content keep only
    the first path in sorted order, and a content found under two classes is
    left out entirely because its label is ambiguous. Paths come back sorted.
    """
    root = find_class_root(root)
    groups = {}
    for label, (folder, _) in enumerate(XRAY_CLASSES):
        images = root / folder / "images"
        if not images.is_dir():
            continue
        for path in sorted(images.iterdir()):
            if path.is_file() and is_readable_image(path):
                groups.setdefault(file_digest(path), []).append((path, label))
    kept = sorted(
        group[0] for group in groups.values() if len({label for _, label in group}) == 1
    )
    return [path for path, _ in kept], [label for _, label in kept]


def sample_per_class(
    labels: list[int], counts: dict[int, int], seed: int, exclude: list[int] = ()
) -> list[int]:
    """Draws a fixed number of indices of each class, without repeats.

    Returns sorted indices with exactly ``counts[c]`` items of each class
    ``c``, never choosing an index listed in ``exclude``. The seed fixes the
    draw. Raises ValueError naming the class when fewer eligible items exist
    than requested.
    """
    banned = set(exclude)
    generator = random.Random(seed)
    chosen = []
    for label, wanted in sorted(counts.items()):
        pool = [i for i, value in enumerate(labels) if value == label and i not in banned]
        if len(pool) < wanted:
            raise ValueError(
                f"class {label} has {len(pool)} eligible images; {wanted} were requested"
            )
        chosen.extend(generator.sample(pool, wanted))
    return sorted(chosen)


def load_grayscale(paths: list[Path], size: int) -> Tensor:
    """Loads images as one uint8 tensor shaped (n, 1, size, size).

    Every file is converted to grayscale and resized to a square with
    antialiasing, whatever its colour mode or original shape.
    """
    arrays = []
    for path in paths:
        with open_image(path) as image:
            resized = image.convert("L").resize((size, size), Resampling.LANCZOS)
            arrays.append(np.asarray(resized, dtype=np.uint8).copy())
    if not arrays:
        return torch.empty((0, 1, size, size), dtype=torch.uint8)
    return torch.from_numpy(np.stack(arrays)).unsqueeze(1)


def class_report(confusion: list[list[int]]) -> list[dict]:
    """Turns a confusion matrix into recall, precision, F1 and support per class.

    Rows are true classes and columns are predicted classes. Results follow
    the class order. A zero denominator gives 0.0 for that metric.
    """
    size = len(confusion)
    report = []
    for position in range(size):
        true_positive = confusion[position][position]
        support = sum(confusion[position])
        predicted = sum(confusion[row][position] for row in range(size))
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted if predicted else 0.0
        total = precision + recall
        f1 = 2 * precision * recall / total if total else 0.0
        report.append(
            {"recall": recall, "precision": precision, "f1": f1, "support": support}
        )
    return report


def mix_synthetic(
    real_images: Tensor,
    real_labels: list[int],
    synthetic_images: Tensor,
    synthetic_label: int,
) -> tuple[Tensor, list[int]]:
    """Builds a training set with synthetic images appended after the real ones.

    Returns new images and labels; the inputs are not modified. Only training
    data should pass through here, so evaluation sets stay purely real.
    Raises ValueError when the dtypes, the number of dimensions or the image
    shape differ, or when the label count does not match the real images, so
    nothing is cast or truncated silently.
    """
    if real_images.dtype != synthetic_images.dtype:
        raise ValueError(
            f"dtype mismatch: real images are {real_images.dtype}, synthetic images are {synthetic_images.dtype}"
        )
    if real_images.dim() != synthetic_images.dim():
        raise ValueError(
            f"dimensions mismatch: real images have {real_images.dim()}, synthetic images have {synthetic_images.dim()}"
        )
    if real_images.shape[1:] != synthetic_images.shape[1:]:
        raise ValueError(
            f"shape mismatch: real images are {tuple(real_images.shape[1:])}, synthetic images are {tuple(synthetic_images.shape[1:])}"
        )
    if len(real_labels) != real_images.shape[0]:
        raise ValueError(
            f"got {len(real_labels)} labels for {real_images.shape[0]} real images"
        )
    images = torch.cat([real_images, synthetic_images], dim=0)
    labels = list(real_labels) + [synthetic_label] * synthetic_images.shape[0]
    return images, labels


def summarise_runs(reports: list[list[dict]], metric: str) -> list[dict]:
    """Summarises one metric per class across runs with different seeds.

    ``reports`` holds one ``class_report`` result per run. Returns, in class
    order, a dict with the class index, the mean and the sample standard
    deviation of the metric. A single run has a standard deviation of 0.0.
    Raises ValueError for no reports, reports with different class counts, or
    a metric name the reports do not hold.
    """
    if not reports:
        raise ValueError("no reports to summarise")
    if len({len(report) for report in reports}) != 1:
        raise ValueError("all reports must have the same number of classes")
    if any(metric not in row for report in reports for row in report):
        raise ValueError(f"unknown metric {metric!r}")
    summary = []
    for position in range(len(reports[0])):
        values = [report[position][metric] for report in reports]
        spread = stdev(values) if len(values) > 1 else 0.0
        summary.append({"class": position, "mean": fmean(values), "std": spread})
    return summary


def inverse_frequency_weights(labels: list[int], num_classes: int) -> Tensor:
    """Returns one loss weight per class, inversely proportional to its share of the labels.

    A class with ``n`` of ``total`` labels gets ``total / (num_classes * n)``,
    so a balanced set gives all ones. Raises ValueError naming the class when
    it has no label, since its weight would be infinite.
    """
    counts = torch.bincount(torch.as_tensor(labels, dtype=torch.long), minlength=num_classes).float()
    missing = (counts == 0).nonzero()
    if missing.numel():
        raise ValueError(f"class {int(missing[0])} has no labels, so it has no weight")
    return counts.sum() / (num_classes * counts)


def build_resnet18(num_classes: int, pretrained: bool) -> nn.Module:
    """Builds a ResNet-18 with a new final layer of ``num_classes`` outputs.

    With ``pretrained`` the backbone starts from the ImageNet weights, and
    otherwise from random ones. Every layer stays trainable.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class XrayBatches:
    """Batches of grayscale X-rays served from uint8 tensors held in memory, ready for a ResNet.

    Iterating yields ``(images, labels)`` with float images shaped
    (n, 3, size, size): the single channel is repeated three times, resized to
    ``output_size`` when given and normalised with the ImageNet statistics.
    With ``augment``, every image first gets its own random rotation of up to
    ``max_degrees`` and its own random shift of up to ``max_shift`` pixels, with
    black filling the corners and no mirroring, since left and right matter in
    a chest X-ray. The random numbers come from a CPU generator seeded by
    ``seed``, so a seed fixes shuffling and augmentation on any device, and
    every new iteration is one epoch.
    """

    def __init__(
        self,
        images: Tensor,
        labels: Tensor,
        batch_size: int,
        output_size: int | None = None,
        augment: bool = False,
        shuffle: bool = False,
        max_shift: float = 4.0,
        max_degrees: float = 7.0,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ):
        """Moves the images and labels to the device and stores the batch settings."""
        self.images = images.to(device)
        self.labels = torch.as_tensor(labels, dtype=torch.long).to(device)
        self.batch_size = batch_size
        self.output_size = output_size
        self.augment = augment
        self.shuffle = shuffle
        self.max_shift = max_shift
        self.max_degrees = max_degrees
        self.generator = torch.Generator().manual_seed(seed)
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def __len__(self) -> int:
        """Returns the number of batches in one epoch."""
        return -(-self.images.shape[0] // self.batch_size)

    def transform(self, images: Tensor) -> Tensor:
        """Rotates and shifts every image of a float batch in 0 to 1 by its own random amounts."""
        count, _, height, width = images.shape
        draws = torch.rand(count, 3, generator=self.generator) * 2 - 1
        angles = draws[:, 0] * self.max_degrees * pi / 180
        shift_x = draws[:, 1] * self.max_shift * 2 / width
        shift_y = draws[:, 2] * self.max_shift * 2 / height
        theta = torch.stack(
            [
                torch.stack([angles.cos(), -angles.sin(), shift_x], dim=1),
                torch.stack([angles.sin(), angles.cos(), shift_y], dim=1),
            ],
            dim=1,
        ).to(images.device)
        grid = affine_grid(theta, list(images.shape), align_corners=False)
        return grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def __iter__(self):
        """Yields the batches of one epoch."""
        count = self.images.shape[0]
        if self.shuffle:
            order = torch.randperm(count, generator=self.generator).to(self.images.device)
        else:
            order = torch.arange(count, device=self.images.device)
        for start in range(0, count, self.batch_size):
            chosen = order[start : start + self.batch_size]
            images = self.images[chosen].float() / 255
            if self.augment:
                images = self.transform(images)
            images = images.expand(-1, 3, -1, -1)
            if self.output_size is not None and self.output_size != images.shape[-1]:
                images = interpolate(
                    images, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False
                )
            yield (images - self.mean) / self.std, self.labels[chosen]
