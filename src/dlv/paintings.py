"""Paintings dataset: year parsing, balanced sampling, painter-wise splitting and image reading."""

import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from random import Random
from unicodedata import combining, normalize

import numpy as np
import torch
from PIL import Image
from pyarrow.compute import struct_field
from pyarrow.parquet import ParquetFile, read_table
from requests import get
from torch import Tensor
from torch.utils.data import Dataset

WIKIART_API_URL = "https://huggingface.co/api/datasets/Artificio/WikiArt"
WIKIART_FILE_URL = "https://huggingface.co/datasets/Artificio/WikiArt/resolve/main/{path}"
PERIODS = [
    ("até 1599", 0, 1599),
    ("1600 a 1799", 1600, 1799),
    ("1800 a 1849", 1800, 1849),
    ("1850 a 1874", 1850, 1874),
    ("1875 a 1899", 1875, 1899),
    ("1900 a 1924", 1900, 1924),
    ("1925 a 1949", 1925, 1949),
    ("1950 em diante", 1950, 9999),
]
YEAR_PATTERN = re.compile(r"(?<!\d)(1\d{3}|20[0-2]\d)(?!\d)")
UNKNOWN_ARTIST_NAMES = {"", "none", "nan", "null", "anonymous", "anonymous artist"}


def parse_year(text) -> int | None:
    """Returns the first four-digit year from 1000 to 2029 found in a text.

    Forms such as ``1920``, ``1997.0`` and ``c.1904`` all yield their year.
    Anything without such a year, including ``None``, yields ``None``.
    """
    match = YEAR_PATTERN.search(str(text)) if text is not None else None
    return int(match.group()) if match else None


def period_of(year: int) -> int:
    """Returns the index into ``PERIODS`` of the period holding a year."""
    for index, (_, first, last) in enumerate(PERIODS):
        if first <= year <= last:
            return index
    raise ValueError(f"year {year} is outside every period")


def download_wikiart(destination: Path) -> list[Path]:
    """Downloads the WikiArt parquet files from Hugging Face, once each.

    The file list comes from the Hugging Face API. Each file is streamed to a
    temporary name and renamed when complete, so an interrupted run never
    leaves a partial parquet file. A ``.complete`` marker written last makes a
    repeated call return immediately. Returns the local paths, sorted.
    """
    destination = Path(destination)
    marker = destination / ".complete"
    if marker.exists():
        return sorted(destination.glob("*.parquet"))
    destination.mkdir(parents=True, exist_ok=True)
    with get(WIKIART_API_URL, timeout=60) as response:
        response.raise_for_status()
        listing = response.json()
    remote_paths = sorted(
        entry["rfilename"]
        for entry in listing["siblings"]
        if entry["rfilename"].startswith("data/") and entry["rfilename"].endswith(".parquet")
    )
    local_paths = []
    for remote_path in remote_paths:
        target = destination / Path(remote_path).name
        local_paths.append(target)
        if target.exists():
            continue
        partial = target.with_suffix(".part")
        with get(WIKIART_FILE_URL.format(path=remote_path), stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(partial, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    handle.write(chunk)
        partial.rename(target)
    marker.touch()
    return sorted(local_paths)


def read_metadata(parquet_paths) -> dict[str, list]:
    """Reads the year, artist, style and title of every painting, without the images.

    Only the ``date``, ``artist``, ``style`` and ``title`` columns are read.
    The result holds parallel lists under ``year`` (int or None), ``artist``,
    ``style``, ``title`` and ``row``, the global row number counted across the
    files in path order.
    """
    metadata = {"year": [], "artist": [], "style": [], "title": [], "row": []}
    offset = 0
    for path in parquet_paths:
        table = read_table(path, columns=["date", "artist", "style", "title"])
        metadata["year"].extend(parse_year(value) for value in table.column("date").to_pylist())
        metadata["artist"].extend(table.column("artist").to_pylist())
        metadata["style"].extend(table.column("style").to_pylist())
        metadata["title"].extend(table.column("title").to_pylist())
        metadata["row"].extend(range(offset, offset + table.num_rows))
        offset += table.num_rows
    return metadata


def row_group_sizes(parquet_paths) -> list[list[int]]:
    """Returns the number of rows of every row group, one list per file.

    ``read_images`` loads one row group at a time, so the largest group sets
    how much memory a read needs. Only the file footers are read.
    """
    sizes = []
    for path in parquet_paths:
        details = ParquetFile(path).metadata
        sizes.append([details.row_group(index).num_rows for index in range(details.num_row_groups)])
    return sizes


def select_balanced(labels: list[int | None], per_class: int, seed: int) -> list[int]:
    """Picks up to ``per_class`` indices of each class, leaving out ``None`` labels.

    The pick is random but fixed by the seed. Classes with fewer rows keep all
    of them. Returns the chosen indices sorted.
    """
    members = defaultdict(list)
    for index, label in enumerate(labels):
        if label is not None:
            members[label].append(index)
    chosen = []
    for label in sorted(members):
        indices = members[label]
        if len(indices) > per_class:
            indices = Random(f"{seed}:{label}").sample(indices, per_class)
        chosen.extend(indices)
    return sorted(chosen)


def is_unknown_artist(name) -> bool:
    """Tells whether an artist value is empty or some spelling of "unknown"."""
    text = str(name).strip().lower() if name is not None else ""
    return text in UNKNOWN_ARTIST_NAMES or text.startswith("unknown")


def spelling_variants(names) -> list[list[str]]:
    """Finds names written in more than one way.

    Two spellings are taken as the same name when they match after ignoring
    case, accents, punctuation, spacing and word order, so ``Monet, Claude``
    and ``claude-monet`` fall together. Returns one sorted list of spellings
    for every name that has at least two, ordered by first spelling. Empty
    names and ``None`` are skipped. Nothing is merged or corrected here.
    """
    spellings = defaultdict(set)
    for name in names:
        text = str(name).strip() if name is not None else ""
        plain = "".join(letter for letter in normalize("NFKD", text) if not combining(letter))
        key = " ".join(sorted(re.sub(r"[^0-9a-z]+", " ", plain.casefold()).split()))
        if key:
            spellings[key].add(text)
    return sorted(sorted(group) for group in spellings.values() if len(group) > 1)


def split_by_group(
    groups: list[str],
    labels: list[int],
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Splits indices into train, validation and test with no group in two parts.

    Whole groups are assigned, largest first, to the part whose class-wise
    quota they fill best, so part sizes follow the fractions and every class
    reaches every part when the data allows. A group that is empty or an
    unknown artist stands for no real painter, so each such row is its own
    group. The result is fixed by the seed. Returns three sorted index lists.
    """
    members = defaultdict(list)
    for index, group in enumerate(groups):
        key = ("row", index) if is_unknown_artist(group) else ("artist", group)
        members[key].append(index)
    classes = sorted(set(labels))
    class_position = {label: position for position, label in enumerate(classes)}
    totals = np.zeros(len(classes))
    for label in labels:
        totals[class_position[label]] += 1
    shares = np.array(fractions) / sum(fractions)
    quota = np.maximum(shares[:, None] * totals[None, :], 1e-9)
    filled = np.zeros_like(quota)
    order = sorted(members, key=repr)
    Random(seed).shuffle(order)
    order.sort(key=lambda key: -len(members[key]))
    parts = ([], [], [])
    for key in order:
        counts = np.zeros(len(classes))
        for index in members[key]:
            counts[class_position[labels[index]]] += 1
        room = (quota - filled) / quota
        scores = (room * counts[None, :]).sum(axis=1) + 1e-6 * room.mean(axis=1)
        part = int(np.argmax(scores))
        parts[part].extend(members[key])
        filled[part] += counts
    return tuple(sorted(part) for part in parts)


def read_images(parquet_paths, rows: list[int]) -> list[bytes]:
    """Returns the encoded image bytes of the given global rows, in the order given.

    Files are opened one at a time and only the ``image`` column of the row
    groups holding wanted rows is read, one row group at a time, so memory
    stays near the size of the selection instead of a whole file.
    """
    wanted = defaultdict(list)
    offset = 0
    spans = []
    for path in parquet_paths:
        parquet_file = ParquetFile(path)
        counts = [parquet_file.metadata.row_group(i).num_rows for i in range(parquet_file.num_row_groups)]
        spans.append((path, parquet_file, offset, counts))
        offset += sum(counts)
    starts = []
    for path, _, file_offset, counts in spans:
        group_offset = file_offset
        for group_index, count in enumerate(counts):
            starts.append((group_offset, group_offset + count, path, group_index))
            group_offset += count
    for row in rows:
        if not 0 <= row < offset:
            raise IndexError(f"row {row} is outside the {offset} rows available")
    ordered = sorted(range(len(rows)), key=lambda position: rows[position])
    cursor = 0
    for first, last, path, group_index in starts:
        while cursor < len(ordered) and rows[ordered[cursor]] < last:
            wanted[(path, group_index)].append((rows[ordered[cursor]] - first, ordered[cursor]))
            cursor += 1
    files = {path: parquet_file for path, parquet_file, _, _ in spans}
    result = [b""] * len(rows)
    for (path, group_index), picks in wanted.items():
        column = files[path].read_row_group(group_index, columns=["image"]).column("image")
        encoded = struct_field(column.combine_chunks(), "bytes").take([local for local, _ in picks])
        for (_, position), data_bytes in zip(picks, encoded.to_pylist()):
            result[position] = data_bytes
    return result


def decode_images(images: list[bytes], size: int) -> Tensor:
    """Decodes encoded images into a uint8 tensor of shape ``(n, 3, size, size)``.

    Every image is converted to RGB and resized to a square with antialiasing.
    """
    arrays = []
    for encoded in images:
        with Image.open(BytesIO(encoded)) as image:
            resized = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC, reducing_gap=None)
            arrays.append(np.asarray(resized))
    if not arrays:
        return torch.empty((0, 3, size, size), dtype=torch.uint8)
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()


class EncodedImageDataset(Dataset):
    """Serves transformed images and labels from encoded image bytes."""

    def __init__(self, images: list[bytes], labels: list[int], transform):
        """Stores the encoded images, their labels and the transform to apply."""
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """Returns the number of images."""
        return len(self.images)

    def __getitem__(self, index: int):
        """Decodes one image as RGB and returns it transformed, with its label."""
        with Image.open(BytesIO(self.images[index])) as image:
            return self.transform(image.convert("RGB")), self.labels[index]


def accuracy_by_group(targets, predictions, groups, minimum: int = 20) -> list[dict]:
    """Summarizes how well each group was predicted.

    Returns one entry per group with at least ``minimum`` samples, holding
    ``group``, ``count``, ``accuracy`` and ``mean_distance`` (the mean absolute
    difference between target and prediction). Entries are sorted by count,
    largest first, and by group name among equal counts.
    """
    gathered = defaultdict(list)
    for target, prediction, group in zip(targets, predictions, groups):
        gathered[group].append((int(target), int(prediction)))
    entries = []
    for group, pairs in gathered.items():
        if len(pairs) < minimum:
            continue
        entries.append(
            {
                "group": group,
                "count": len(pairs),
                "accuracy": sum(t == p for t, p in pairs) / len(pairs),
                "mean_distance": sum(abs(t - p) for t, p in pairs) / len(pairs),
            }
        )
    return sorted(entries, key=lambda entry: (-entry["count"], str(entry["group"])))


def evenly_spread(items: list, count: int) -> list:
    """Picks ``count`` items at even steps from first to last, or all when there are fewer."""
    if len(items) <= count:
        return list(items)
    if count == 1:
        return [items[0]]
    return [items[round(step * (len(items) - 1) / (count - 1))] for step in range(count)]


def select_examples(targets, predictions, per_outcome: int = 3) -> list[int]:
    """Chooses samples to show: correct predictions first, then wrong ones.

    For each outcome up to ``per_outcome`` samples are taken from classes
    spread evenly over the class order, the first sample of each class. Wrong
    predictions come from classes not used by the correct ones when enough
    remain, so the examples cover as many classes as possible. When an outcome
    has too few classes, its next samples in order fill the gap, and the list
    is shorter when an outcome has too few samples. The choice depends only on
    the inputs. Returns the indices, each outcome ordered by class.
    """
    targets = [int(target) for target in targets]
    predictions = [int(prediction) for prediction in predictions]
    used_classes = set()
    chosen = []
    for correct in (True, False):
        candidates = [
            index for index, (target, prediction) in enumerate(zip(targets, predictions))
            if (target == prediction) == correct
        ]
        first_of_class = {}
        for index in candidates:
            first_of_class.setdefault(targets[index], index)
        classes = sorted(first_of_class)
        unused = [label for label in classes if label not in used_classes]
        pool = unused if len(unused) >= per_outcome else classes
        picked = [first_of_class[label] for label in evenly_spread(pool, per_outcome)]
        picked.extend([index for index in candidates if index not in picked][: per_outcome - len(picked)])
        used_classes.update(targets[index] for index in picked)
        chosen.extend(sorted(picked, key=lambda index: (targets[index], index)))
    return chosen
