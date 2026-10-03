"""Plots used by the notebooks and the report. Labels are in Portuguese."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


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
    axis.bar(class_names, per_class_accuracy)
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("Classe")
    axis.set_ylabel("Acurácia")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    return figure
