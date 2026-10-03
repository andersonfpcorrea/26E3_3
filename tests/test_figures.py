"""Tests for the plotting helpers."""

from dlv.figures import plot_confusion, plot_history, plot_per_class_accuracy

HISTORY = {
    "train_loss": [1.0, 0.5],
    "train_accuracy": [0.5, 0.8],
    "val_loss": [1.1, 0.7],
    "val_accuracy": [0.4, 0.7],
}


def test_plot_history_saves_loss_and_accuracy_panels(tmp_path):
    figure = plot_history(HISTORY, tmp_path / "history.png")
    assert (tmp_path / "history.png").stat().st_size > 0
    assert [axis.get_ylabel() for axis in figure.axes] == ["Perda", "Acurácia"]


def test_plot_confusion_labels_axes_with_class_names(tmp_path):
    figure = plot_confusion([[3, 1], [0, 4]], ["bike", "car"], tmp_path / "confusion.png")
    axis = figure.axes[0]
    assert (tmp_path / "confusion.png").stat().st_size > 0
    assert [label.get_text() for label in axis.get_xticklabels()] == ["bike", "car"]
    assert axis.get_ylabel() == "Classe real"


def test_plot_per_class_accuracy_draws_one_bar_per_class(tmp_path):
    figure = plot_per_class_accuracy([0.75, 1.0], ["bike", "car"], tmp_path / "per_class.png")
    assert (tmp_path / "per_class.png").stat().st_size > 0
    assert len(figure.axes[0].patches) == 2
