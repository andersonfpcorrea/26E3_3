"""Tests for the X-ray data module."""

from math import isclose

import pytest
import torch
from PIL import Image

from dlv.xray import (
    XRAY_CLASSES,
    XrayBatches,
    build_resnet18,
    class_report,
    count_xray_files,
    inverse_frequency_weights,
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
    assert summarise_runs([report], "recall") == [{"class": 0, "mean": 0.4, "std": 0.0, "values": [0.4]}]


def test_listing_finds_class_folders_under_wrapper_parent(tmp_path):
    """A root above the wrapper folder still finds the class folders."""
    wrapper = make_xray_tree(tmp_path / "COVID-19_Radiography_Dataset")
    paths, labels = list_xray_images(tmp_path)
    assert len(paths) == 9
    assert all(wrapper in path.parents for path in paths)


def test_listing_finds_class_folders_when_root_is_wrapper(tmp_path):
    """The wrapper folder itself works as root."""
    wrapper = make_xray_tree(tmp_path / "COVID-19_Radiography_Dataset")
    paths, _ = list_xray_images(wrapper)
    assert len(paths) == 9


def test_listing_uses_shallowest_candidate(tmp_path):
    """With several candidate folders the shallowest one wins."""
    make_xray_tree(tmp_path / "a", per_class=2)
    make_xray_tree(tmp_path / "b" / "deeper", per_class=3)
    paths, _ = list_xray_images(tmp_path)
    assert len(paths) == 6


def test_listing_raises_when_no_class_folder(tmp_path):
    """An empty root raises an error naming root and the layout."""
    with pytest.raises(ValueError, match=str(tmp_path)) as error:
        list_xray_images(tmp_path)
    assert "Normal/images" in str(error.value)


def test_mix_synthetic_rejects_dtype_mismatch():
    """A float synthetic tensor is not cast to uint8."""
    real = torch.zeros((2, 1, 4, 4), dtype=torch.uint8)
    fake = torch.zeros((2, 1, 4, 4), dtype=torch.float32)
    with pytest.raises(ValueError, match="dtype"):
        mix_synthetic(real, [0, 1], fake, 2)


def test_mix_synthetic_rejects_dimension_mismatch():
    """Different numbers of dimensions are rejected."""
    real = torch.zeros((2, 1, 4, 4), dtype=torch.uint8)
    fake = torch.zeros((2, 4, 4), dtype=torch.uint8)
    with pytest.raises(ValueError, match="dimensions"):
        mix_synthetic(real, [0, 1], fake, 2)


def test_mix_synthetic_rejects_shape_mismatch():
    """Different image shapes are rejected."""
    real = torch.zeros((2, 1, 4, 4), dtype=torch.uint8)
    fake = torch.zeros((2, 1, 8, 8), dtype=torch.uint8)
    with pytest.raises(ValueError, match="shape"):
        mix_synthetic(real, [0, 1], fake, 2)


def test_mix_synthetic_rejects_label_count_mismatch():
    """The label list must match the real images."""
    real = torch.zeros((2, 1, 4, 4), dtype=torch.uint8)
    fake = torch.zeros((1, 1, 4, 4), dtype=torch.uint8)
    with pytest.raises(ValueError, match="labels"):
        mix_synthetic(real, [0, 1, 1], fake, 2)


def test_summarise_runs_rejects_empty_reports():
    """No reports raise a clear error."""
    with pytest.raises(ValueError, match="no reports"):
        summarise_runs([], "recall")


def test_summarise_runs_rejects_different_lengths():
    """Reports with different class counts raise."""
    row = {"recall": 0.5, "precision": 0.5, "f1": 0.5, "support": 1}
    with pytest.raises(ValueError, match="same number of classes"):
        summarise_runs([[row, row], [row]], "recall")


def test_summarise_runs_rejects_unknown_metric():
    """An unknown metric name raises and names it."""
    row = {"recall": 0.5, "precision": 0.5, "f1": 0.5, "support": 1}
    with pytest.raises(ValueError, match="accuracy"):
        summarise_runs([[row]], "accuracy")


def test_sampling_second_draw_excluding_first_does_not_overlap():
    """A test set drawn with exclude never overlaps the first draw."""
    labels = [0] * 20 + [1] * 20
    first = sample_per_class(labels, {0: 5, 1: 3}, seed=1)
    second = sample_per_class(labels, {0: 7, 1: 9}, seed=2, exclude=first)
    assert not set(first) & set(second)
    assert [labels[i] for i in second].count(0) == 7
    assert [labels[i] for i in second].count(1) == 9
    assert [labels[i] for i in first].count(0) == 5
    assert [labels[i] for i in first].count(1) == 3


def test_count_xray_files_counts_copies_and_unreadable_files_but_not_masks(tmp_path):
    """The count covers every image-named file of the three classes before cleaning."""
    make_xray_tree(tmp_path)
    save(tmp_path / "Normal" / "images" / "copy.png", 17)
    (tmp_path / "Normal" / "images" / "broken.png").write_bytes(b"not an image")
    assert count_xray_files(tmp_path) == 11


def test_inverse_frequency_weights_balance_the_classes():
    """A balanced set gives ones and a rarer class gets a larger weight."""
    assert inverse_frequency_weights([0, 1, 2, 0, 1, 2], 3).tolist() == [1.0, 1.0, 1.0]
    weights = inverse_frequency_weights([0, 0, 0, 0, 1, 2], 3)
    assert weights.tolist() == pytest.approx([0.5, 2.0, 2.0])


def test_inverse_frequency_weights_name_a_class_without_labels():
    """A class that never appears has no finite weight, so the error names it."""
    with pytest.raises(ValueError, match="class 2"):
        inverse_frequency_weights([0, 1, 1], 3)


def test_build_resnet18_has_a_new_head_and_trainable_layers():
    """The model ends in one output per class and every parameter trains."""
    model = build_resnet18(3, pretrained=False)
    assert model.fc.out_features == 3
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert model(torch.zeros(2, 3, 64, 64)).shape == (2, 3)


def xray_batches(images, **overrides):
    """Builds batches over uint8 images with labels counting up from zero."""
    settings = dict(images=images, labels=torch.arange(images.shape[0]), batch_size=4)
    settings.update(overrides)
    return XrayBatches(**settings)


def test_xray_batches_repeat_channels_resize_and_normalise():
    """Batches are float, three channels, resized and shifted by the ImageNet statistics."""
    images = torch.full((5, 1, 8, 8), 255, dtype=torch.uint8)
    batches = list(xray_batches(images, output_size=16))
    assert [batch.shape[0] for batch, _ in batches] == [4, 1]
    assert batches[0][0].shape == (4, 3, 16, 16)
    assert batches[0][0][0, 0, 0, 0].item() == pytest.approx((1 - 0.485) / 0.229, abs=1e-4)
    assert batches[0][0][0, 2, 0, 0].item() == pytest.approx((1 - 0.406) / 0.225, abs=1e-4)
    assert len(xray_batches(images)) == 2


def test_xray_batches_without_options_keep_order_and_labels():
    """Without shuffling or augmentation the same images come back in order, unchanged."""
    images = torch.randint(0, 256, (6, 1, 8, 8), dtype=torch.uint8)
    first = list(xray_batches(images))
    second = list(xray_batches(images))
    labels = torch.cat([batch_labels for _, batch_labels in first])
    assert labels.tolist() == list(range(6))
    assert torch.equal(first[0][0], second[0][0])


def test_xray_batches_augmentation_moves_pixels_without_mirroring():
    """Augmented images differ from plain ones, change between epochs and a seed repeats them."""
    images = torch.zeros(8, 1, 16, 16, dtype=torch.uint8)
    images[:, :, 4:12, 2:6] = 200
    plain = next(iter(xray_batches(images, batch_size=8)))[0]
    augmented = xray_batches(images, batch_size=8, augment=True, seed=1)
    epoch_one = next(iter(augmented))[0]
    epoch_two = next(iter(augmented))[0]
    repeated = next(iter(xray_batches(images, batch_size=8, augment=True, seed=1)))[0]
    assert not torch.equal(plain, epoch_one)
    assert not torch.equal(epoch_one, epoch_two)
    assert torch.equal(epoch_one, repeated)
    left, right = epoch_one[:, 0, :, :8], epoch_one[:, 0, :, 8:]
    assert left.sum() > 2 * right.sum()


def test_xray_batches_zero_range_augmentation_is_the_identity():
    """With no shift and no rotation allowed, augmentation leaves the image as it is."""
    images = torch.randint(0, 256, (4, 1, 8, 8), dtype=torch.uint8)
    plain = next(iter(xray_batches(images)))[0]
    same = next(iter(xray_batches(images, augment=True, max_shift=0.0, max_degrees=0.0)))[0]
    assert torch.allclose(plain, same, atol=1e-4)


def test_xray_batches_shuffle_visits_every_image_once():
    """Shuffling changes the order but every label appears exactly once per epoch."""
    images = torch.zeros(10, 1, 8, 8, dtype=torch.uint8)
    batches = xray_batches(images, shuffle=True, seed=3)
    labels = torch.cat([batch_labels for _, batch_labels in batches]).tolist()
    assert sorted(labels) == list(range(10))
    assert labels != list(range(10))


def test_summarise_runs_keeps_the_value_of_every_run():
    """Each summary row lists the metric of every run in report order."""
    reports = [[{"recall": value, "precision": 0.0, "f1": 0.0, "support": 1}] for value in (0.5, 0.7, 0.9)]
    assert summarise_runs(reports, "recall")[0]["values"] == [0.5, 0.7, 0.9]
