"""Paintings dataset: year parsing, balanced sampling, painter-wise splitting and image reading."""

import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from random import Random

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
    """Reads the year, artist and style of every painting, without the images.

    Only the ``date``, ``artist`` and ``style`` columns are read. The result
    holds parallel lists under ``year`` (int or None), ``artist``, ``style``
    and ``row``, the global row number counted across the files in path order.
    """
    metadata = {"year": [], "artist": [], "style": [], "row": []}
    offset = 0
    for path in parquet_paths:
        table = read_table(path, columns=["date", "artist", "style"])
        metadata["year"].extend(parse_year(value) for value in table.column("date").to_pylist())
        metadata["artist"].extend(table.column("artist").to_pylist())
        metadata["style"].extend(table.column("style").to_pylist())
        metadata["row"].extend(range(offset, offset + table.num_rows))
        offset += table.num_rows
    return metadata


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
