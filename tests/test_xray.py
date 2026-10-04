"""Tests for the X-ray data module."""

from math import isclose

import pytest
import torch
from PIL import Image

from dlv.xray import (
    XRAY_CLASSES,
    class_report,
    list_xray_images,
    load_grayscale,
    mix_synthetic,
    sample_per_class,
    summarise_runs,
)


def save(path, value, mode="L", size=(8, 8)):
    """Saves a solid image, creating the parent folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    colour = value if mode == "L" else tuple([value] * len(mode))
    Image.new(mode, size, colour).save(path)


def make_xray_tree(root, per_class=3):
    """Builds class folders with images and masks plus a class to be ignored."""
    value = 10
    for folder, _ in XRAY_CLASSES:
        for index in range(per_class):
            value += 7
            save(root / folder / "images" / f"{folder}-{index}.png", value)
            save(root / folder / "masks" / f"{folder}-{index}.png", value + 1)
    for index in range(per_class):
        save(root / "Lung_Opacity" / "images" / f"lo-{index}.png", 200 + index)
    return root


def test_listing_ignores_masks_and_other_classes(tmp_path):
    """Only images of the three classes are listed."""
    paths, labels = list_xray_images(make_xray_tree(tmp_path))
    assert len(paths) == 9
    assert all(path.parent.name == "images" for path in paths)
    assert all("Lung_Opacity" not in path.parts for path in paths)
    assert paths == sorted(paths)


def test_listing_labels_by_class_folder(tmp_path):
    """The label is the index of the folder in XRAY_CLASSES."""
    paths, labels = list_xray_images(make_xray_tree(tmp_path))
    for path, label in zip(paths, labels):
        assert path.parent.parent.name == XRAY_CLASSES[label][0]
    assert sorted(set(labels)) == [0, 1, 2]


def test_listing_drops_exact_copies_keeping_first(tmp_path):
    """A copy within a class is removed and the first sorted path stays."""
    make_xray_tree(tmp_path)
    save(tmp_path / "Normal" / "images" / "Normal-0.png", 77)
    save(tmp_path / "Normal" / "images" / "zz-copy.png", 77)
    paths, _ = list_xray_images(tmp_path)
    names = [path.name for path in paths]
    assert "Normal-0.png" in names
    assert "zz-copy.png" not in names
    assert len(paths) == 9


def test_listing_drops_cross_class_copies(tmp_path):
    """A content found under two classes is left out of both."""
    make_xray_tree(tmp_path)
    save(tmp_path / "Normal" / "images" / "shared.png", 150)
    save(tmp_path / "COVID" / "images" / "shared.png", 150)
    paths, _ = list_xray_images(tmp_path)
    assert "shared.png" not in [path.name for path in paths]
    assert len(paths) == 9


def test_listing_skips_unreadable_files(tmp_path):
    """A file that is not a decodable image is not listed."""
    make_xray_tree(tmp_path)
    (tmp_path / "Normal" / "images" / "broken.png").write_bytes(b"not an image")
    paths, _ = list_xray_images(tmp_path)
    assert "broken.png" not in [path.name for path in paths]


def test_sampling_counts_and_sorted():
    """Each class gets exactly the requested count, in sorted order."""
    labels = [0] * 10 + [1] * 8 + [2] * 6
    chosen = sample_per_class(labels, {0: 4, 1: 3, 2: 2}, seed=1)
    assert chosen == sorted(chosen)
    assert [labels[i] for i in chosen].count(0) == 4
    assert [labels[i] for i in chosen].count(1) == 3
    assert [labels[i] for i in chosen].count(2) == 2


def test_sampling_is_deterministic_and_seed_dependent():
    """The same seed gives the same draw and another seed another one."""
    labels = [0] * 30 + [1] * 30
    first = sample_per_class(labels, {0: 5, 1: 5}, seed=3)
    assert first == sample_per_class(labels, {0: 5, 1: 5}, seed=3)
    assert first != sample_per_class(labels, {0: 5, 1: 5}, seed=4)


def test_sampling_never_picks_excluded():
    """Excluded indices are never chosen, and the rest can be exhausted."""
    labels = [0] * 10 + [1] * 10
    excluded = list(range(0, 6)) + list(range(10, 16))
    chosen = sample_per_class(labels, {0: 4, 1: 4}, seed=0, exclude=excluded)
    assert not set(chosen) & set(excluded)
    assert chosen == [6, 7, 8, 9, 16, 17, 18, 19]


def test_sampling_error_names_the_class():
    """Asking for more than exists raises and names the class."""
    labels = [0] * 5 + [1] * 2
    with pytest.raises(ValueError, match="class 1"):
        sample_per_class(labels, {0: 3, 1: 3}, seed=0)
    with pytest.raises(ValueError, match="class 0"):
        sample_per_class(labels, {0: 3}, seed=0, exclude=[0, 1, 2])


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_load_grayscale_shape_and_dtype(tmp_path, mode):
    """Any colour mode comes out as uint8 (n, 1, size, size)."""
    paths = []
    for index in range(3):
        path = tmp_path / f"{index}.png"
        save(path, 50 * (index + 1), mode=mode, size=(20, 12))
        paths.append(path)
    tensor = load_grayscale(paths, 8)
    assert tensor.shape == (3, 1, 8, 8)
    assert tensor.dtype == torch.uint8
    assert tensor[2].float().mean() > tensor[0].float().mean()


def test_load_grayscale_empty_list():
    """No paths give an empty tensor of the right shape."""
    assert load_grayscale([], 8).shape == (0, 1, 8, 8)


def test_class_report_hand_computed():
    """Metrics match values worked out by hand, including a never predicted class."""
    confusion = [[8, 2, 0], [1, 3, 0], [2, 2, 0]]
    report = class_report(confusion)
    assert [row["support"] for row in report] == [10, 4, 4]
    assert isclose(report[0]["recall"], 0.8)
    assert isclose(report[0]["precision"], 8 / 11)
    assert isclose(report[0]["f1"], 2 * 0.8 * (8 / 11) / (0.8 + 8 / 11))
    assert isclose(report[1]["recall"], 0.75)
    assert isclose(report[1]["precision"], 3 / 7)
    assert report[2] == {"recall": 0.0, "precision": 0.0, "f1": 0.0, "support": 4}


def test_class_report_empty_class_support():
    """A class with no true samples gets zeros instead of a division error."""
    report = class_report([[5, 0], [0, 0]])
    assert report[1] == {"recall": 0.0, "precision": 0.0, "f1": 0.0, "support": 0}


def test_mix_synthetic_shapes_labels_and_inputs_untouched():
    """Synthetic images are appended with their label and inputs stay as they were."""
    real = torch.randint(0, 256, (5, 1, 4, 4), dtype=torch.uint8)
    fake = torch.randint(0, 256, (3, 1, 4, 4), dtype=torch.uint8)
    real_copy, fake_copy = real.clone(), fake.clone()
    labels = [0, 0, 1, 1, 2]
    images, mixed = mix_synthetic(real, labels, fake, 2)
    assert images.shape == (8, 1, 4, 4)
    assert images.dtype == torch.uint8
    assert mixed == [0, 0, 1, 1, 2, 2, 2, 2]
    assert torch.equal(images[:5], real_copy)
    assert torch.equal(images[5:], fake_copy)
    assert torch.equal(real, real_copy)
    assert torch.equal(fake, fake_copy)
    assert labels == [0, 0, 1, 1, 2]


def test_summarise_runs_mean_and_std():
    """Mean and sample standard deviation match hand-computed values."""
    reports = [
        [{"recall": 0.5, "precision": 0.1, "f1": 0.2, "support": 4}, {"recall": 1.0, "precision": 0.0, "f1": 0.0, "support": 2}],
        [{"recall": 0.7, "precision": 0.3, "f1": 0.4, "support": 4}, {"recall": 0.0, "precision": 0.0, "f1": 0.0, "support": 2}],
        [{"recall": 0.9, "precision": 0.2, "f1": 0.3, "support": 4}, {"recall": 0.5, "precision": 0.0, "f1": 0.0, "support": 2}],
    ]
    summary = summarise_runs(reports, "recall")
    assert [row["class"] for row in summary] == [0, 1]
    assert isclose(summary[0]["mean"], 0.7)
    assert isclose(summary[0]["std"], 0.2)
    assert isclose(summary[1]["mean"], 0.5)
    assert isclose(summary[1]["std"], 0.5)
    assert isclose(summarise_runs(reports, "f1")[0]["std"], 0.1)


def test_summarise_runs_single_run_has_zero_std():
    """One run has nothing to vary against."""
    report = [{"recall": 0.4, "precision": 0.4, "f1": 0.4, "support": 1}]
    assert summarise_runs([report], "recall") == [{"class": 0, "mean": 0.4, "std": 0.0}]
