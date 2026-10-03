"""Tests for dataset download, listing and splitting."""

from collections import Counter
from io import BytesIO
from shutil import copytree
from zipfile import ZipFile, BadZipFile

import pytest
import torch
from PIL import Image
from requests import HTTPError
from torchvision.transforms import CenterCrop, Normalize, Resize, ToTensor

from dlv import data
from dlv.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ImageListDataset,
    count_image_files,
    download_kaggle_dataset,
    imagenet_transform,
    list_image_folder,
    stratified_split,
)


def make_image_tree(root, classes=("bike", "car"), per_class=10):
    """Creates a folder per class with small solid-colour images."""
    for position, name in enumerate(classes):
        folder = root / "data" / name
        folder.mkdir(parents=True)
        for index in range(per_class):
            Image.new("RGB", (8, 8), (index * 20, position * 100, 0)).save(folder / f"{name}_{index}.bmp")
    return root


def test_list_image_folder_leaves_out_exact_copies(tmp_path):
    make_image_tree(tmp_path)
    copytree(tmp_path / "data", tmp_path / "data" / "data")
    paths, labels, _ = list_image_folder(tmp_path)
    assert len(paths) == 20
    assert Counter(labels) == {0: 10, 1: 10}
    contents = [path.read_bytes() for path in paths]
    assert len(set(contents)) == len(contents)


def test_list_image_folder_drops_images_shared_by_two_classes(tmp_path):
    make_image_tree(tmp_path)
    original = tmp_path / "data" / "bike" / "bike_3.bmp"
    copy = tmp_path / "data" / "car" / "shared.bmp"
    copy.write_bytes(original.read_bytes())
    paths, labels, _ = list_image_folder(tmp_path)
    assert original not in paths
    assert copy not in paths
    assert Counter(labels) == {0: 9, 1: 10}


class FakeResponse:
    """Stands in for a streamed HTTP response carrying a zip archive."""

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *details):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield self.payload


def zip_payload():
    """Returns the bytes of a zip archive holding one tiny image."""
    image = BytesIO()
    Image.new("RGB", (4, 4)).save(image, format="PNG")
    buffer = BytesIO()
    with ZipFile(buffer, "w") as bundle:
        bundle.writestr("data/bike/one.png", image.getvalue())
    return buffer.getvalue()


def test_download_extracts_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "get", lambda *args, **kwargs: FakeResponse(zip_payload()))
    destination = download_kaggle_dataset("owner/name", tmp_path / "set")
    assert (destination / "data" / "bike" / "one.png").exists()
    assert not (destination / "archive.zip").exists()


def test_download_is_skipped_when_already_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "get", lambda *args, **kwargs: FakeResponse(zip_payload()))
    download_kaggle_dataset("owner/name", tmp_path / "set")

    def fail(*args, **kwargs):
        raise AssertionError("downloaded twice")

    monkeypatch.setattr(data, "get", fail)
    assert download_kaggle_dataset("owner/name", tmp_path / "set") == tmp_path / "set"


def test_download_propagates_http_error(tmp_path, monkeypatch):
    """Download failure does not create completion marker."""
    class FailingResponse:
        def __enter__(self):
            return self

        def __exit__(self, *details):
            return False

        def raise_for_status(self):
            raise HTTPError("404 Not Found")

        def iter_content(self, chunk_size):
            yield b""

    monkeypatch.setattr(data, "get", lambda *args, **kwargs: FailingResponse())
    with pytest.raises(HTTPError):
        download_kaggle_dataset("owner/name", tmp_path / "set")
    assert not (tmp_path / "set" / ".complete").exists()


def test_download_propagates_bad_zip_error(tmp_path, monkeypatch):
    """Invalid zip payload does not create completion marker."""
    monkeypatch.setattr(data, "get", lambda *args, **kwargs: FakeResponse(b"<html>not a zip</html>"))
    with pytest.raises(BadZipFile):
        download_kaggle_dataset("owner/name", tmp_path / "set")
    assert not (tmp_path / "set" / ".complete").exists()


def test_list_image_folder_labels_by_parent_folder(tmp_path):
    make_image_tree(tmp_path)
    paths, labels, class_names = list_image_folder(tmp_path)
    assert class_names == ["bike", "car"]
    assert len(paths) == 20
    assert Counter(labels) == {0: 10, 1: 10}
    assert all(class_names[label] == path.parent.name for path, label in zip(paths, labels))


def test_list_image_folder_skips_unreadable_and_non_image_files(tmp_path):
    make_image_tree(tmp_path)
    (tmp_path / "data" / "bike" / "broken.jpg").write_bytes(b"not an image")
    (tmp_path / "data" / "bike" / "notes.txt").write_text("hello")
    (tmp_path / "data" / "bike" / ".hidden.png").write_bytes(b"x")
    paths, _, _ = list_image_folder(tmp_path)
    assert len(paths) == 20
    assert all(path.suffix == ".bmp" for path in paths)


def test_list_image_folder_skips_truncated_images(tmp_path):
    """Skips truncated BMP files that cannot be decoded during actual use."""
    make_image_tree(tmp_path)
    full_bmp_path = tmp_path / "data" / "bike" / "full.bmp"
    Image.new("RGB", (64, 64), (100, 100, 100)).save(full_bmp_path)
    full_bytes = full_bmp_path.read_bytes()
    truncated_bmp_path = tmp_path / "data" / "bike" / "truncated.bmp"
    truncated_bmp_path.write_bytes(full_bytes[:len(full_bytes) // 2])
    paths, _, _ = list_image_folder(tmp_path)
    assert truncated_bmp_path not in paths
    assert len(paths) == 21


def test_stratified_split_keeps_proportions_and_is_disjoint():
    labels = [0] * 100 + [1] * 40
    train, val, test = stratified_split(labels)
    assert sorted(train + val + test) == list(range(140))
    assert Counter(labels[i] for i in train) == {0: 70, 1: 28}
    assert Counter(labels[i] for i in val) == {0: 15, 1: 6}
    assert Counter(labels[i] for i in test) == {0: 15, 1: 6}


def test_stratified_split_is_deterministic():
    labels = [0] * 50 + [1] * 50
    assert stratified_split(labels, seed=7) == stratified_split(labels, seed=7)


def test_stratified_split_names_the_class_that_is_too_small():
    with pytest.raises(ValueError, match="class 1 has 3 images"):
        stratified_split([0] * 50 + [1] * 3)


def test_image_list_dataset_returns_tensor_and_label(tmp_path):
    make_image_tree(tmp_path)
    paths, labels, _ = list_image_folder(tmp_path)
    dataset = ImageListDataset(paths, labels, ToTensor())
    image, label = dataset[0]
    assert len(dataset) == 20
    assert image.shape == (3, 8, 8)
    assert image.dtype == torch.float32
    assert label == labels[0]


def test_imagenet_transform_produces_normalized_square_tensor():
    """A white photo becomes a 224x224 tensor scaled by the ImageNet statistics."""
    tensor = imagenet_transform()(Image.new("RGB", (640, 480), (255, 255, 255)))
    assert tensor.shape == (3, 224, 224)
    assert tensor.max() > 2.0


def test_imagenet_transform_defaults_match_the_resnet50_v2_preset():
    """By default the short side is resized to 232 and the centre crop is 224."""
    resize, crop, _, normalize = imagenet_transform().transforms
    assert isinstance(resize, Resize)
    assert resize.size == 232
    assert isinstance(crop, CenterCrop)
    assert tuple(crop.size) == (224, 224)
    assert isinstance(normalize, Normalize)
    assert tuple(normalize.mean) == IMAGENET_MEAN == (0.485, 0.456, 0.406)
    assert tuple(normalize.std) == IMAGENET_STD == (0.229, 0.224, 0.225)


def test_imagenet_transform_respects_custom_sizes():
    """Other resize and crop sizes are used as given."""
    transform = imagenet_transform(resize_size=256, crop_size=192)
    resize, crop = transform.transforms[:2]
    assert resize.size == 256
    assert tuple(crop.size) == (192, 192)
    assert transform(Image.new("RGB", (640, 480))).shape == (3, 192, 192)


def test_count_image_files_counts_visible_files_with_image_suffixes(tmp_path):
    """Copies, unreadable images and upper-case suffixes count; hidden and other files do not."""
    make_image_tree(tmp_path)
    copytree(tmp_path / "data", tmp_path / "data" / "data")
    (tmp_path / "data" / "bike" / "broken.JPG").write_bytes(b"not an image")
    (tmp_path / "data" / "bike" / "notes.txt").write_text("hello")
    (tmp_path / "data" / "bike" / ".hidden.png").write_bytes(b"x")
    (tmp_path / "data" / "folder.png").mkdir()
    assert count_image_files(tmp_path) == 41


def test_count_image_files_is_zero_for_a_missing_folder(tmp_path):
    """A folder that does not exist holds no images."""
    assert count_image_files(tmp_path / "missing") == 0
