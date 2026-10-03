"""Dataset download, image discovery, splitting and preprocessing shared by the notebooks."""

from collections import Counter
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
from requests import get
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

KAGGLE_DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/{slug}"
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png"}
MINIMUM_IMAGES_PER_CLASS = 7
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def download_kaggle_dataset(slug: str, destination: Path) -> Path:
    """Downloads and extracts a public Kaggle dataset, once.

    A marker file records a finished download, so calling this again for the
    same destination returns immediately. No Kaggle account is needed.
    """
    destination = Path(destination)
    marker = destination / ".complete"
    if marker.exists():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "archive.zip"
    with get(KAGGLE_DOWNLOAD_URL.format(slug=slug), stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(archive, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    with ZipFile(archive) as bundle:
        bundle.extractall(destination)
    archive.unlink()
    marker.touch()
    return destination


def has_image_name(path: Path) -> bool:
    """Tells whether a path is named like a visible image file."""
    return not path.name.startswith(".") and path.suffix.lower() in IMAGE_SUFFIXES


def count_image_files(root: Path) -> int:
    """Counts the visible files under root that are named like images.

    Nothing is opened, so exact copies and files that cannot be decoded are
    counted too. This is the size of a download before any cleaning.
    """
    return sum(1 for path in Path(root).rglob("*") if path.is_file() and has_image_name(path))


def is_readable_image(path: Path) -> bool:
    """Tells whether a file is a visible image that Pillow can decode."""
    if not has_image_name(path):
        return False
    try:
        with Image.open(path) as image:
            image.load()
    except (OSError, SyntaxError, ValueError):
        return False
    return True


def file_digest(path: Path) -> str:
    """Returns the SHA-256 hash of a file's bytes."""
    return sha256(path.read_bytes()).hexdigest()


def list_image_folder(root: Path) -> tuple[list[Path], list[int], list[str]]:
    """Finds every readable image under root and labels it by its parent folder.

    Returns the image paths, one integer label per path, and the class names
    in the order used by the labels. Unreadable files are left out. Exact
    copies within one class keep only the first path in sorted order. Files
    with identical content under two or more classes have an ambiguous label,
    so all of them are left out.
    """
    groups = {}
    for path in sorted(path for path in Path(root).rglob("*") if path.is_file() and is_readable_image(path)):
        groups.setdefault(file_digest(path), []).append(path)
    paths = sorted(
        group[0] for group in groups.values() if len({path.parent.name for path in group}) == 1
    )
    class_names = sorted({path.parent.name for path in paths})
    index = {name: position for position, name in enumerate(class_names)}
    labels = [index[path.parent.name] for path in paths]
    return paths, labels, class_names


def stratified_split(
    labels: list[int],
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Splits indices into train, validation and test, keeping class shares.

    Raises ValueError naming the class when a class is too small to appear in
    all three parts.
    """
    for label, count in sorted(Counter(labels).items()):
        if count < MINIMUM_IMAGES_PER_CLASS:
            raise ValueError(
                f"class {label} has {count} images; at least {MINIMUM_IMAGES_PER_CLASS} are needed to split"
            )
    indices = list(range(len(labels)))
    train, rest = train_test_split(
        indices, train_size=fractions[0], stratify=labels, random_state=seed
    )
    validation_share = fractions[1] / (fractions[1] + fractions[2])
    validation, test = train_test_split(
        rest,
        train_size=validation_share,
        stratify=[labels[i] for i in rest],
        random_state=seed,
    )
    return sorted(train), sorted(validation), sorted(test)


class ImageListDataset(Dataset):
    """Serves transformed images and labels from parallel lists."""

    def __init__(self, paths: list[Path], labels: list[int], transform):
        """Stores the image paths, their labels and the transform to apply."""
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """Returns the number of images."""
        return len(self.paths)

    def __getitem__(self, index: int):
        """Loads one image as RGB and returns it transformed, with its label."""
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB")), self.labels[index]


def imagenet_transform(resize_size: int = 232, crop_size: int = 224) -> Compose:
    """Returns the evaluation preprocessing of a backbone pre-trained on ImageNet.

    The short side is resized to ``resize_size``, a centre square of
    ``crop_size`` is cut out and the channels are normalized with the ImageNet
    mean and standard deviation. The defaults are the published preset of the
    ResNet-50 ``IMAGENET1K_V2`` weights. Other weights may use other sizes.
    """
    return Compose(
        [
            Resize(resize_size),
            CenterCrop(crop_size),
            ToTensor(),
            Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
