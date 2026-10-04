"""Plots used by the notebooks and the report. Labels are in Portuguese."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor
from torch.nn.functional import interpolate


def plot_history(history: dict, path: Path) -> Figure:
    """Draws loss and accuracy per epoch for training and validation."""
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(11, 4))
    loss_axis.plot(epochs, history["train_loss"], marker="o", label="Treino")
    loss_axis.plot(epochs, history["val_loss"], marker="o", label="Validação")
    loss_axis.set_xlabel("Época")
    loss_axis.set_ylabel("Perda")
    loss_axis.legend()
    accuracy_axis.plot(epochs, history["train_accuracy"], marker="o", label="Treino")
    accuracy_axis.plot(epochs, history["val_accuracy"], marker="o", label="Validação")
    accuracy_axis.set_xlabel("Época")
    accuracy_axis.set_ylabel("Acurácia")
    accuracy_axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    return figure


def plot_confusion(confusion: list[list[int]], class_names: list[str], path: Path) -> Figure:
    """Draws a confusion matrix with true classes on rows."""
    size = max(5, 0.7 * len(class_names))
    figure, axis = plt.subplots(figsize=(size, size))
    axis.imshow(confusion, cmap="Blues")
    positions = range(len(class_names))
    axis.set_xticks(positions, labels=class_names, rotation=45, ha="right")
    axis.set_yticks(positions, labels=class_names)
    axis.set_xlabel("Classe prevista")
    axis.set_ylabel("Classe real")
    limit = max(max(row) for row in confusion) / 2
    for row_index, row in enumerate(confusion):
        for column_index, value in enumerate(row):
            colour = "white" if value > limit else "black"
            axis.text(column_index, row_index, str(value), ha="center", va="center", color=colour)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    return figure


def plot_per_class_accuracy(
    per_class_accuracy: list[float], class_names: list[str], path: Path
) -> Figure:
    """Draws one bar per class with its accuracy on the test set."""
    figure, axis = plt.subplots(figsize=(max(6, 0.8 * len(class_names)), 4))
    positions = range(len(class_names))
    axis.bar(positions, per_class_accuracy)
    axis.set_xticks(positions, labels=class_names, rotation=45, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("Classe")
    axis.set_ylabel("Acurácia")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    return figure


def plot_attention_maps(
    images: Tensor,
    maps: Tensor,
    titles: list[str],
    path: Path,
    panel_titles: list[str] | None = None,
    title: str | None = None,
) -> Figure:
    """Draws each image next to its attention maps, one panel per head.

    `images` is a uint8 tensor (n, 3, h, w) and `maps` a float tensor (n, heads, g, g).
    Each map is upsampled to the image size and drawn semi-transparent over the
    painting, so the image stays visible under it. `titles` has one entry per row.
    `panel_titles` names the map columns and defaults to the head numbers.
    Every panel is colour-scaled on its own, from its lowest to its highest
    value, so colours cannot be compared between panels; `title` is the place
    to tell the reader.
    """
    count, heads = maps.shape[0], maps.shape[1]
    height, width = images.shape[-2:]
    upsampled = interpolate(maps.float(), size=(height, width), mode="bilinear", align_corners=False)
    figure, axes = plt.subplots(
        count, heads + 1, figsize=(2.2 * (heads + 1), 2.5 * count), squeeze=False
    )
    for row in range(count):
        painting = images[row].permute(1, 2, 0).numpy()
        axes[row][0].imshow(painting)
        axes[row][0].set_title(titles[row], fontsize=9)
        for head in range(heads):
            axis = axes[row][head + 1]
            axis.imshow(painting)
            axis.imshow(upsampled[row, head].numpy(), cmap="jet", alpha=0.5)
            if row == 0:
                name = panel_titles[head] if panel_titles else f"Cabeça {head + 1}"
                axis.set_title(name, fontsize=9)
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    if title:
        figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    return figure


def plot_group_accuracy(entries: list[dict], path: Path, title: str) -> Figure:
    """Draws horizontal bars of accuracy per group, with the sample count on each bar.

    `entries` is the output of `accuracy_by_group`; the first entry is drawn at the top.
    """
    figure, axis = plt.subplots(figsize=(7, max(2.5, 0.45 * len(entries) + 1.5)))
    positions = range(len(entries))
    accuracies = [entry["accuracy"] for entry in entries]
    axis.barh(positions, accuracies)
    axis.set_yticks(positions, labels=[entry["group"] for entry in entries])
    axis.invert_yaxis()
    axis.set_xlim(0, 1.3)
    axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    for position, entry in zip(positions, entries):
        axis.text(
            entry["accuracy"] + 0.01,
            position,
            f"{entry['accuracy']:.0%} (n={entry['count']})",
            va="center",
            fontsize=8,
        )
    axis.set_xlabel("Acurácia")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    return figure


def plot_histories(histories: dict[str, dict], path: Path) -> Figure:
    """Draws loss and accuracy of several named runs on shared axes.

    Loss shows training (dashed) and validation (solid) curves; accuracy shows validation only.
    Each run keeps one colour in both panels and may have its own number of epochs.
    """
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(11, 4))
    for index, (name, history) in enumerate(histories.items()):
        colour = f"C{index}"
        epochs = range(1, len(history["train_loss"]) + 1)
        loss_axis.plot(
            epochs, history["train_loss"], linestyle="--", color=colour, label=f"{name} (treino)"
        )
        loss_axis.plot(
            epochs, history["val_loss"], marker="o", color=colour, label=f"{name} (validação)"
        )
        val_epochs = range(1, len(history["val_accuracy"]) + 1)
        accuracy_axis.plot(
            val_epochs, history["val_accuracy"], marker="o", color=colour, label=name
        )
    loss_axis.set_xlabel("Época")
    loss_axis.set_ylabel("Perda")
    loss_axis.legend(fontsize=8)
    accuracy_axis.set_xlabel("Época")
    accuracy_axis.set_ylabel("Acurácia")
    accuracy_axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    return figure


def plot_class_counts(counts: dict[str, dict[str, int]], path: Path) -> Figure:
    """Draws grouped bars of samples per class, one bar for each split.

    `counts` maps a split name to a mapping from class name to sample count.
    """
    splits = list(counts)
    class_names = list(counts[splits[0]])
    width = 0.8 / len(splits)
    figure, axis = plt.subplots(figsize=(max(6, 0.9 * len(class_names)), 4))
    for split_index, split in enumerate(splits):
        offsets = [
            class_index + (split_index - (len(splits) - 1) / 2) * width
            for class_index in range(len(class_names))
        ]
        axis.bar(offsets, [counts[split][name] for name in class_names], width, label=split)
    axis.set_xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right")
    axis.set_xlabel("Classe")
    axis.set_ylabel("Número de amostras")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    return figure


def plot_probability_distributions(
    top_probabilities: Tensor, all_probabilities: Tensor, threshold: float, path: Path
) -> Figure:
    """Draws two histograms of concept probabilities with the presence threshold marked.

    The left panel holds the highest probability of each image, the right one
    every probability of every image, on a logarithmic count axis.
    """
    figure, (top_axis, all_axis) = plt.subplots(1, 2, figsize=(11, 4))
    top_axis.hist(top_probabilities.flatten().tolist(), bins=40, range=(0, 1))
    top_axis.set_xlabel("Maior probabilidade da imagem")
    top_axis.set_ylabel("Número de imagens")
    top_axis.set_title("Conceito mais provável de cada imagem")
    all_axis.hist(all_probabilities.flatten().tolist(), bins=40, range=(0, 1))
    all_axis.set_yscale("log")
    all_axis.set_xlabel("Probabilidade")
    all_axis.set_ylabel("Número de pares imagem e conceito (escala log)")
    all_axis.set_title("Todos os conceitos de todas as imagens")
    for axis in (top_axis, all_axis):
        axis.axvline(threshold, color="C3", linestyle="--", label=f"Limiar {threshold}")
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    return figure


def plot_ranking(rankings: dict[str, list[dict]], path: Path, top: int = 15) -> Figure:
    """Draws horizontal frequency bars of the best concepts, one panel for each named ranking.

    Each ranking is the output of ``rank_concepts``; the most frequent concept
    is on top and the mean similarity is written beside each bar.
    """
    figure, axes = plt.subplots(1, len(rankings), figsize=(6 * len(rankings), 6), squeeze=False)
    for axis, (name, ranking) in zip(axes[0], rankings.items()):
        rows = ranking[:top]
        positions = range(len(rows))
        axis.barh(positions, [row["frequency"] for row in rows])
        axis.set_yticks(positions, labels=[row["concept"].removeprefix("a photo of ") for row in rows], fontsize=8)
        axis.invert_yaxis()
        largest = max([row["frequency"] for row in rows], default=0.0)
        axis.set_xlim(0, largest * 1.45 + 1e-9)
        for position, row in zip(positions, rows):
            axis.text(
                row["frequency"], position,
                f" {row['frequency']:.1%} (sim. {row['mean_similarity']:.3f})",
                va="center", fontsize=7,
            )
        axis.set_xlabel("Frequência")
        axis.set_title(name)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    return figure


def plot_image_grid(
    paths: list[Path], captions: list[str], title: str, path: Path, columns: int = 5
) -> Figure:
    """Draws images in a grid of ``columns`` columns, each with its caption, under one title."""
    rows = max(1, -(-len(paths) // columns))
    figure, axes = plt.subplots(rows, columns, figsize=(2.4 * columns, 2.6 * rows + 0.6), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, image_path, caption in zip(axes.ravel(), paths, captions):
        with Image.open(image_path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(caption, fontsize=7)
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    return figure
