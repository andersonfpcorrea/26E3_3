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
    TensorBatches,
    channel_statistics,
    count_image_files,
    download_kaggle_dataset,
    imagenet_transform,
    list_image_folder,
    random_crop_flip,
    stratified_split,
)
from dlv.engine import evaluate, fit


def make_image_tree(root, classes=("bike", "car"), per_class=10):
    """Creates a folder per class with small solid-colour images."""
    for position, name in enumerate(classes):
        folder = root / "data" / name
        folder.mkdir(parents=True)
        for index in range(per_class):
            Image.new("RGB", (8, 8), (index * 20, position * 100, 0)).save(folder / f"{name}_{index}.bmp")
    return root


def test_list_image_folder_leaves_out_exact_copies(tmp_path):
    """A download holding the dataset twice yields each image once."""
    make_image_tree(tmp_path)
    copytree(tmp_path / "data", tmp_path / "data" / "data")
    paths, labels, _ = list_image_folder(tmp_path)
    assert len(paths) == 20
    assert Counter(labels) == {0: 10, 1: 10}
    contents = [path.read_bytes() for path in paths]
    assert len(set(contents)) == len(contents)


def test_list_image_folder_drops_images_shared_by_two_classes(tmp_path):
    """An image whose copy sits in another class is dropped from both."""
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
        """Stores the bytes the response will serve."""
        self.payload = payload

    def __enter__(self):
        """Enters the context and returns the response itself."""
        return self

    def __exit__(self, *details):
        """Leaves the context without swallowing errors."""
        return False

    def raise_for_status(self):
        """Reports a successful status."""
        return None

    def iter_content(self, chunk_size):
        """Serves the whole payload as one chunk."""
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
    """The archive is extracted into the destination and then deleted."""
    monkeypatch.setattr(data, "get", lambda *args, **kwargs: FakeResponse(zip_payload()))
    destination = download_kaggle_dataset("owner/name", tmp_path / "set")
    assert (destination / "data" / "bike" / "one.png").exists()
    assert not (destination / "archive.zip").exists()


def test_download_is_skipped_when_already_complete(tmp_path, monkeypatch):
    """A finished download is not fetched again."""
    monkeypatch.setattr(data, "get", lambda *args, **kwargs: FakeResponse(zip_payload()))
    download_kaggle_dataset("owner/name", tmp_path / "set")

    def fail(*args, **kwargs):
        """Fails the test if a second download is attempted."""
        raise AssertionError("downloaded twice")

    monkeypatch.setattr(data, "get", fail)
    assert download_kaggle_dataset("owner/name", tmp_path / "set") == tmp_path / "set"


def test_download_propagates_http_error(tmp_path, monkeypatch):
    """Download failure does not create completion marker."""
    class FailingResponse:
        """Stands in for a response with an HTTP error status."""
        def __enter__(self):
            """Enters the context and returns the response itself."""
            return self

        def __exit__(self, *details):
            """Leaves the context without swallowing errors."""
            return False

        def raise_for_status(self):
            """Raises the HTTP error of a missing dataset."""
            raise HTTPError("404 Not Found")

        def iter_content(self, chunk_size):
            """Serves one empty chunk."""
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
    """Each image is labelled with the name of its parent folder."""
    make_image_tree(tmp_path)
    paths, labels, class_names = list_image_folder(tmp_path)
    assert class_names == ["bike", "car"]
    assert len(paths) == 20
    assert Counter(labels) == {0: 10, 1: 10}
    assert all(class_names[label] == path.parent.name for path, label in zip(paths, labels))


def test_list_image_folder_skips_unreadable_and_non_image_files(tmp_path):
    """Broken images, text files and hidden files are left out."""
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
    """The three parts cover every index once and keep the class shares."""
    labels = [0] * 100 + [1] * 40
    train, val, test = stratified_split(labels)
    assert sorted(train + val + test) == list(range(140))
    assert Counter(labels[i] for i in train) == {0: 70, 1: 28}
    assert Counter(labels[i] for i in val) == {0: 15, 1: 6}
    assert Counter(labels[i] for i in test) == {0: 15, 1: 6}


def test_stratified_split_is_deterministic():
    """The same seed gives the same split."""
    labels = [0] * 50 + [1] * 50
    assert stratified_split(labels, seed=7) == stratified_split(labels, seed=7)


def test_stratified_split_names_the_class_that_is_too_small():
    """A class too small to split is named in the error."""
    with pytest.raises(ValueError, match="class 1 has 3 images"):
        stratified_split([0] * 50 + [1] * 3)


def test_image_list_dataset_returns_tensor_and_label(tmp_path):
    """The dataset serves a float tensor and the matching label."""
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


def numbered_images(count=6, size=8):
    """Returns uint8 images whose pixels all differ, so any move is visible."""
    return torch.arange(count * 3 * size * size).remainder(251).to(torch.uint8).reshape(count, 3, size, size)


def test_random_crop_flip_keeps_shape_and_dtype():
    """The augmented batch has the shape and type of the input."""
    images = numbered_images()
    augmented = random_crop_flip(images, padding=2, generator=torch.Generator().manual_seed(0))
    assert augmented.shape == images.shape
    assert augmented.dtype == torch.uint8


def test_random_crop_flip_without_padding_only_mirrors():
    """With no padding every image comes back as itself or as its mirror."""
    images = numbered_images(count=40)
    augmented = random_crop_flip(images, padding=0, generator=torch.Generator().manual_seed(0))
    same = [torch.equal(new, old) for new, old in zip(augmented, images)]
    mirrored = [torch.equal(new, old.flip(-1)) for new, old in zip(augmented, images)]
    assert all(a or b for a, b in zip(same, mirrored))
    assert any(same) and any(mirrored)


def test_random_crop_flip_shifts_by_at_most_the_padding():
    """Each output is a window of the zero-padded image or of its mirror."""
    images = numbered_images(count=30)
    padding = 2
    augmented = random_crop_flip(images, padding=padding, generator=torch.Generator().manual_seed(1))
    padded = torch.nn.functional.pad(images, (padding,) * 4)
    size = images.shape[-1]
    offsets = set()
    for index, new in enumerate(augmented):
        matches = [
            (top, left)
            for top in range(2 * padding + 1)
            for left in range(2 * padding + 1)
            for source in (padded[index], padded[index].flip(-1))
            if torch.equal(new, source[:, top : top + size, left : left + size])
        ]
        assert matches
        offsets.add(matches[0])
    assert len(offsets) > 5


def test_random_crop_flip_is_fixed_by_the_generator():
    """The same generator seed gives the same augmented batch."""
    images = numbered_images()
    first = random_crop_flip(images, padding=2, generator=torch.Generator().manual_seed(3))
    second = random_crop_flip(images, padding=2, generator=torch.Generator().manual_seed(3))
    assert torch.equal(first, second)


def test_channel_statistics_gives_mean_and_std_on_the_unit_scale():
    """Mean and standard deviation are per channel, with pixels scaled to 0..1."""
    images = torch.zeros(2, 3, 4, 4, dtype=torch.uint8)
    images[:, 0] = 255
    images[0, 1] = 255
    mean, std = channel_statistics(images)
    assert mean == pytest.approx((1.0, 0.5, 0.0))
    assert std == pytest.approx((0.0, 0.5, 0.0))


def test_tensor_batches_serves_every_sample_normalized_and_in_order():
    """Without shuffling, batches follow the stored order and are normalized floats."""
    images = numbered_images(count=5)
    labels = torch.arange(5)
    batches = TensorBatches(images, labels, batch_size=2, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))
    served = list(batches)
    assert len(batches) == 3
    assert [batch_images.shape[0] for batch_images, _ in served] == [2, 2, 1]
    assert torch.equal(torch.cat([batch_labels for _, batch_labels in served]), labels)
    expected = (images.float() / 255 - 0.5) / 0.25
    assert torch.allclose(torch.cat([batch_images for batch_images, _ in served]), expected)
    assert served[0][0].dtype == torch.float32


def test_tensor_batches_shuffles_differently_each_epoch_but_keeps_pairs():
    """Shuffled epochs differ in order while every image keeps its label."""
    images = torch.arange(20, dtype=torch.uint8).reshape(20, 1, 1, 1).expand(20, 3, 2, 2).contiguous()
    labels = torch.arange(20)
    batches = TensorBatches(images, labels, batch_size=4, mean=(0.0,) * 3, std=(1.0,) * 3, shuffle=True, seed=1)
    orders = []
    for _ in range(2):
        epoch_labels = []
        for batch_images, batch_labels in batches:
            assert torch.equal((batch_images[:, 0, 0, 0] * 255).round().long(), batch_labels)
            epoch_labels.extend(batch_labels.tolist())
        assert sorted(epoch_labels) == list(range(20))
        orders.append(epoch_labels)
    assert orders[0] != orders[1]
    assert orders[0] != list(range(20))


def test_tensor_batches_is_repeatable_for_a_seed():
    """Two instances with the same seed serve the same augmented batches."""
    images = numbered_images(count=6)
    labels = torch.arange(6)

    def first_batch():
        """Returns the first batch of a fresh shuffled, augmented instance."""
        batches = TensorBatches(
            images, labels, batch_size=4, mean=(0.0,) * 3, std=(1.0,) * 3, augment=True, shuffle=True, seed=5
        )
        return next(iter(batches))

    assert torch.equal(first_batch()[0], first_batch()[0])
    assert torch.equal(first_batch()[1], first_batch()[1])


def test_tensor_batches_augments_only_when_asked():
    """Augmentation changes the pixels served; without it they are untouched."""
    images = numbered_images(count=16)
    labels = torch.arange(16)
    plain = TensorBatches(images, labels, batch_size=16, mean=(0.0,) * 3, std=(1.0,) * 3)
    augmented = TensorBatches(images, labels, batch_size=16, mean=(0.0,) * 3, std=(1.0,) * 3, augment=True)
    assert torch.allclose(next(iter(plain))[0], images.float() / 255)
    assert not torch.allclose(next(iter(augmented))[0], images.float() / 255)


def test_tensor_batches_works_with_the_training_loop():
    """The engine trains and evaluates from in-memory batches."""
    torch.manual_seed(0)
    images = torch.cat([torch.full((8, 3, 4, 4), 20), torch.full((8, 3, 4, 4), 230)]).to(torch.uint8)
    labels = torch.cat([torch.zeros(8, dtype=torch.long), torch.ones(8, dtype=torch.long)])
    batches = TensorBatches(images, labels, batch_size=4, mean=(0.5,) * 3, std=(0.5,) * 3, shuffle=True)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(48, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    history = fit(model, batches, batches, optimizer, epochs=3, device=torch.device("cpu"))
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert evaluate(model, batches, torch.device("cpu"), num_classes=2)["accuracy"] == 1.0
