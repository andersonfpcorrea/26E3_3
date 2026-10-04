"""Tests for the plotting helpers."""

import pytest
import torch
from PIL import Image

from dlv.figures import (
    plot_arm_metric,
    plot_attention_maps,
    plot_class_counts,
    plot_confusion,
    plot_gan_curves,
    plot_gan_diversity,
    plot_gan_snapshots,
    plot_group_accuracy,
    plot_histories,
    plot_history,
    plot_image_grid,
    plot_per_class_accuracy,
    plot_probability_distributions,
    plot_ranking,
    plot_sample_rows,
)

HISTORY = {
    "train_loss": [1.0, 0.5],
    "train_accuracy": [0.5, 0.8],
    "val_loss": [1.1, 0.7],
    "val_accuracy": [0.4, 0.7],
}


def test_plot_history_saves_loss_and_accuracy_panels(tmp_path):
    """The history figure is saved and has a loss panel and an accuracy panel."""
    figure = plot_history(HISTORY, tmp_path / "history.png")
    assert (tmp_path / "history.png").stat().st_size > 0
    assert [axis.get_ylabel() for axis in figure.axes] == ["Perda", "Acurácia"]


def test_plot_confusion_labels_axes_with_class_names(tmp_path):
    """The confusion figure is saved with class names on the x axis."""
    figure = plot_confusion([[3, 1], [0, 4]], ["bike", "car"], tmp_path / "confusion.png")
    axis = figure.axes[0]
    assert (tmp_path / "confusion.png").stat().st_size > 0
    assert [label.get_text() for label in axis.get_xticklabels()] == ["bike", "car"]
    assert axis.get_ylabel() == "Classe real"


def test_plot_per_class_accuracy_draws_one_bar_per_class(tmp_path):
    """The per-class figure is saved with one bar for each class."""
    figure = plot_per_class_accuracy([0.75, 1.0], ["bike", "car"], tmp_path / "per_class.png")
    assert (tmp_path / "per_class.png").stat().st_size > 0
    assert len(figure.axes[0].patches) == 2


def test_plot_per_class_accuracy_anchors_rotated_labels_on_the_right(tmp_path):
    """Class names are rotated and end under their own bar."""
    figure = plot_per_class_accuracy([0.75, 1.0], ["bike", "car"], tmp_path / "per_class.png")
    labels = figure.axes[0].get_xticklabels()
    assert [label.get_text() for label in labels] == ["bike", "car"]
    assert all(label.get_ha() == "right" for label in labels)
    assert all(label.get_rotation() == 45 for label in labels)


def test_plot_attention_maps_draws_image_and_one_panel_per_head(tmp_path):
    """Each row has the image plus one panel per head, and the file is saved."""
    images = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)
    maps = torch.rand(2, 4, 8, 8)
    figure = plot_attention_maps(images, maps, ["a", "b"], tmp_path / "attention.png")
    assert (tmp_path / "attention.png").stat().st_size > 0
    assert len(figure.axes) == 2 * (1 + 4)


def test_plot_attention_maps_titles_rows_and_heads(tmp_path):
    """Row titles label the image panels and head panels are numbered."""
    images = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)
    maps = torch.rand(2, 3, 8, 8)
    figure = plot_attention_maps(images, maps, ["Monet", "Goya"], tmp_path / "attention.png")
    assert [figure.axes[0].get_title(), figure.axes[4].get_title()] == ["Monet", "Goya"]
    assert [axis.get_title() for axis in figure.axes[1:4]] == ["Cabeça 1", "Cabeça 2", "Cabeça 3"]


def test_plot_attention_maps_overlays_map_at_image_size(tmp_path):
    """The overlay is upsampled to the image size and the painting stays underneath."""
    images = torch.randint(0, 256, (1, 3, 32, 32), dtype=torch.uint8)
    maps = torch.rand(1, 2, 8, 8)
    figure = plot_attention_maps(images, maps, ["a"], tmp_path / "attention.png")
    axis = figure.axes[1]
    assert len(axis.images) == 2
    assert axis.images[0].get_array().shape[:2] == (32, 32)
    assert axis.images[1].get_array().shape == (32, 32)
    assert 0 < axis.images[1].get_alpha() < 1


def test_plot_attention_maps_handles_a_single_image(tmp_path):
    """One image still produces a row of panels."""
    images = torch.randint(0, 256, (1, 3, 16, 16), dtype=torch.uint8)
    maps = torch.rand(1, 2, 4, 4)
    figure = plot_attention_maps(images, maps, ["a"], tmp_path / "attention.png")
    assert len(figure.axes) == 3


def test_plot_attention_maps_accepts_panel_titles_and_a_figure_title(tmp_path):
    """Custom panel titles replace the head numbers and the figure gets its own title."""
    images = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)
    maps = torch.rand(2, 1, 4, 4)
    figure = plot_attention_maps(
        images,
        maps,
        ["a", "b"],
        tmp_path / "attention.png",
        panel_titles=["Média das cabeças"],
        title="Cada painel tem sua própria escala de cores",
    )
    assert figure.axes[1].get_title() == "Média das cabeças"
    assert figure.get_suptitle() == "Cada painel tem sua própria escala de cores"


def test_plot_attention_maps_has_no_figure_title_by_default(tmp_path):
    """Without a title argument the figure carries no overall title."""
    images = torch.randint(0, 256, (1, 3, 16, 16), dtype=torch.uint8)
    figure = plot_attention_maps(images, torch.rand(1, 2, 4, 4), ["a"], tmp_path / "attention.png")
    assert figure.get_suptitle() == ""


ENTRIES = [
    {"group": "Impressionismo", "count": 120, "accuracy": 0.8, "mean_distance": 0.5},
    {"group": "Barroco", "count": 40, "accuracy": 0.5, "mean_distance": 1.2},
]


def test_plot_group_accuracy_draws_one_bar_per_group_with_counts(tmp_path):
    """Each group gets a bar and a label showing its count."""
    figure = plot_group_accuracy(ENTRIES, tmp_path / "groups.png", "Por estilo")
    axis = figure.axes[0]
    assert (tmp_path / "groups.png").stat().st_size > 0
    assert len(axis.patches) == 2
    assert axis.get_title() == "Por estilo"
    assert axis.get_xlabel() == "Acurácia"
    assert [label.get_text() for label in axis.get_yticklabels()] == ["Impressionismo", "Barroco"]
    texts = [text.get_text() for text in axis.texts]
    assert any("120" in text for text in texts)
    assert any("40" in text for text in texts)


def test_plot_group_accuracy_has_no_tick_past_one(tmp_path):
    """The axis leaves room for the labels but its ticks stop at 100%."""
    figure = plot_group_accuracy(ENTRIES, tmp_path / "groups.png", "Por estilo")
    axis = figure.axes[0]
    assert list(axis.get_xticks()) == [0, 0.25, 0.5, 0.75, 1.0]
    assert axis.get_xlim()[1] > 1.0


def test_plot_group_accuracy_lists_largest_group_on_top(tmp_path):
    """The first entry is drawn at the top of the chart."""
    figure = plot_group_accuracy(ENTRIES, tmp_path / "groups.png", "Por estilo")
    assert figure.axes[0].yaxis_inverted()


def test_plot_histories_draws_loss_and_accuracy_for_each_run(tmp_path):
    """Loss shows both curves per run and accuracy shows validation only."""
    histories = {"CNN": HISTORY, "ViT": {**HISTORY, "train_loss": [1.0, 0.6, 0.4],
        "val_loss": [1.2, 0.9, 0.8], "train_accuracy": [0.3, 0.5, 0.6], "val_accuracy": [0.3, 0.4, 0.5]}}
    figure = plot_histories(histories, tmp_path / "histories.png")
    loss_axis, accuracy_axis = figure.axes
    assert (tmp_path / "histories.png").stat().st_size > 0
    assert [loss_axis.get_ylabel(), accuracy_axis.get_ylabel()] == ["Perda", "Acurácia"]
    assert len(loss_axis.lines) == 4
    assert len(accuracy_axis.lines) == 2
    labels = [text.get_text() for text in loss_axis.get_legend().get_texts()]
    assert "CNN (treino)" in labels and "ViT (validação)" in labels
    assert [text.get_text() for text in accuracy_axis.get_legend().get_texts()] == ["CNN", "ViT"]


COUNTS = {
    "treino": {"bike": 10, "car": 20},
    "validação": {"bike": 5, "car": 4},
    "teste": {"bike": 6, "car": 7},
}


def test_plot_class_counts_draws_one_bar_per_class_and_split(tmp_path):
    """Grouped bars cover every class for every split, labelled by class."""
    figure = plot_class_counts(COUNTS, tmp_path / "counts.png")
    axis = figure.axes[0]
    assert (tmp_path / "counts.png").stat().st_size > 0
    assert len(axis.patches) == 6
    assert [label.get_text() for label in axis.get_xticklabels()] == ["bike", "car"]
    assert [text.get_text() for text in axis.get_legend().get_texts()] == list(COUNTS)
    assert axis.get_ylabel() == "Número de amostras"


def test_plot_probability_distributions_marks_the_threshold_in_both_panels(tmp_path):
    """Two histograms are saved and each one carries a line at the threshold."""
    probabilities = torch.tensor([[0.7, 0.2, 0.1], [0.4, 0.4, 0.2]])
    figure = plot_probability_distributions(
        probabilities.max(dim=1).values, probabilities, 0.5, tmp_path / "distribution.png"
    )
    assert (tmp_path / "distribution.png").stat().st_size > 0
    assert len(figure.axes) == 2
    for axis in figure.axes:
        assert [line.get_xdata()[0] for line in axis.lines] == [0.5]
        assert [text.get_text() for text in axis.get_legend().get_texts()] == ["Limiar 0.5"]


RANKINGS = {
    "corpus": [
        {"concept": "a photo of a car", "frequency": 0.4, "mean_similarity": 0.21},
        {"concept": "a photo of food", "frequency": 0.1, "mean_similarity": 0.19},
    ],
    "anuncio": [{"concept": "a photo of food", "frequency": 0.3, "mean_similarity": 0.2}],
}


def test_plot_ranking_draws_one_panel_per_ranking_with_the_best_concept_on_top(tmp_path):
    """Each ranking gets a panel titled by its name, with one bar per concept and the prompt prefix removed."""
    figure = plot_ranking(RANKINGS, tmp_path / "ranking.png")
    assert (tmp_path / "ranking.png").stat().st_size > 0
    assert [axis.get_title() for axis in figure.axes] == ["corpus", "anuncio"]
    assert [len(axis.patches) for axis in figure.axes] == [2, 1]
    assert [label.get_text() for label in figure.axes[0].get_yticklabels()] == ["a car", "food"]
    assert figure.axes[0].yaxis_inverted()


def test_plot_ranking_keeps_only_the_top_concepts(tmp_path):
    """Only the first ``top`` concepts of a ranking are drawn."""
    figure = plot_ranking(RANKINGS, tmp_path / "ranking.png", top=1)
    assert [len(axis.patches) for axis in figure.axes] == [1, 1]


def test_plot_image_grid_titles_and_captions_every_image(tmp_path):
    """The grid shows the title, one captioned image per path and hides unused cells."""
    paths = []
    for number in range(3):
        paths.append(tmp_path / f"{number}.png")
        Image.new("RGB", (6, 4), (number * 50, 0, 0)).save(paths[-1])
    figure = plot_image_grid(paths, ["a", "b", "c"], "consulta", tmp_path / "grid.png", columns=2)
    assert (tmp_path / "grid.png").stat().st_size > 0
    assert figure.get_suptitle() == "consulta"
    assert len(figure.axes) == 4
    assert [axis.get_title() for axis in figure.axes[:3]] == ["a", "b", "c"]
    assert len([axis for axis in figure.axes if axis.images]) == 3


GAN_HISTORY = {
    "discriminator_loss": [1.0, 0.8, 0.7],
    "generator_loss": [0.9, 1.1, 1.3],
    "real_score": [0.6, 0.7, 0.7],
    "fake_score": [0.4, 0.3, 0.3],
    "diversity": {0: [0.1, 0.2, 0.2], 1: [0.1, 0.15, 0.2]},
    "snapshots": {1: torch.zeros(1, 20, 34, dtype=torch.uint8), 3: torch.full((1, 20, 34), 200, dtype=torch.uint8)},
}


def test_plot_gan_curves_has_a_loss_and_a_score_panel_with_every_version(tmp_path):
    """Loss and score panels carry the discriminator and generator curves of every version."""
    figure = plot_gan_curves({"v1": GAN_HISTORY, "v2": GAN_HISTORY}, tmp_path / "curves.png")
    assert (tmp_path / "curves.png").stat().st_size > 0
    assert len(figure.axes) == 2
    assert len(figure.axes[0].lines) == 4
    assert len(figure.axes[1].lines) == 4


def test_plot_gan_diversity_draws_every_seed_and_the_mean_for_each_version(tmp_path):
    """Each class panel holds, per version, one thin curve per seed and a thick mean curve, plus the real line, from epoch 2."""
    seeds = [GAN_HISTORY["diversity"], {0: [0.1, 0.3, 0.4], 1: [0.1, 0.25, 0.2]}]
    figure = plot_gan_diversity(
        {"v1": seeds, "v3": seeds},
        {0: 0.3, 1: 0.25},
        ["Normal", "COVID-19"],
        tmp_path / "diversity.png",
    )
    assert len(figure.axes) == 2
    assert figure.axes[0].get_title() == "Normal"
    assert len(figure.axes[0].lines) == 2 * (2 + 1) + 1
    assert list(figure.axes[0].lines[0].get_xdata()) == [2, 3]
    mean_line = figure.axes[0].lines[2]
    assert list(mean_line.get_ydata()) == pytest.approx([0.25, 0.3])
    assert len({line.get_color() for line in figure.axes[0].lines[:3]}) == 1


def test_plot_gan_snapshots_draws_one_panel_per_epoch(tmp_path):
    """Snapshots are drawn in epoch order, each panel titled with its epoch."""
    figure = plot_gan_snapshots(GAN_HISTORY["snapshots"], tmp_path / "snapshots.png", "Titulo")
    assert (tmp_path / "snapshots.png").stat().st_size > 0
    assert [axis.get_title() for axis in figure.axes] == ["Época 1", "Época 3"]


def test_plot_sample_rows_draws_one_labelled_row_per_group(tmp_path):
    """Every named group of images gets one row of panels, labelled with its name."""
    rows = {"reais": torch.zeros(3, 1, 8, 8, dtype=torch.uint8), "geradas": torch.ones(3, 1, 8, 8, dtype=torch.uint8)}
    figure = plot_sample_rows(rows, tmp_path / "rows.png")
    assert (tmp_path / "rows.png").stat().st_size > 0
    assert len(figure.axes) == 6
    assert figure.axes[0].get_ylabel() == "reais"
    assert figure.axes[3].get_ylabel() == "geradas"


def test_plot_arm_metric_draws_a_bar_per_class_and_arm_with_error_bars(tmp_path):
    """Bars sit at the class means of every arm and the labels name the classes."""
    summaries = {
        "A": [{"class": 0, "mean": 0.9, "std": 0.01}, {"class": 1, "mean": 0.5, "std": 0.1}],
        "B": [{"class": 0, "mean": 0.8, "std": 0.02}, {"class": 1, "mean": 0.6, "std": 0.2}],
    }
    figure = plot_arm_metric(summaries, ["Normal", "COVID-19"], "Titulo", tmp_path / "arms.png")
    axis = figure.axes[0]
    heights = sorted(round(patch.get_height(), 2) for patch in axis.patches)
    assert heights == [0.5, 0.6, 0.8, 0.9]
    assert [label.get_text() for label in axis.get_xticklabels()] == ["Normal", "COVID-19"]
    assert axis.get_title() == "Titulo"
