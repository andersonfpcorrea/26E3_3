"""Chest X-ray data preparation and per-class metrics for the COVID-19 screening case study."""

import random
from pathlib import Path
from statistics import fmean, stdev

import numpy as np
import torch
from PIL.Image import Resampling
from PIL.Image import open as open_image
from torch import Tensor

from dlv.data import file_digest, is_readable_image

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
