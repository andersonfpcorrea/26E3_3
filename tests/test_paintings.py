"""Tests for the paintings data module."""

from collections import Counter
from io import BytesIO

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image
from requests import HTTPError
from torchvision.transforms import ToTensor

from dlv import paintings
from dlv.paintings import (
    PERIODS,
    EncodedImageDataset,
    accuracy_by_group,
    decode_images,
    download_wikiart,
    parse_year,
    period_of,
    read_images,
    read_metadata,
    row_group_sizes,
    select_balanced,
    select_examples,
    spelling_variants,
    split_by_group,
)


def encoded_image(value, size=8):
    """Returns PNG bytes of a solid grey image of the given brightness."""
    buffer = BytesIO()
    Image.new("RGB", (size, size), (value, value, value)).save(buffer, format="PNG")
    return buffer.getvalue()


def write_parquet(path, dates, artists, styles, first_value, row_group_size):
    """Writes a parquet file shaped like WikiArt with an image struct column."""
    count = len(dates)
    images = [{"bytes": encoded_image(first_value + i), "path": f"{i}.png"} for i in range(count)]
    table = pa.table(
        {
            "title": [f"t{i}" for i in range(count)],
            "artist": artists,
            "date": dates,
            "style": styles,
            "image": pa.array(
                images, type=pa.struct([("bytes", pa.binary()), ("path", pa.string())])
            ),
        }
    )
    pq.write_table(table, path, row_group_size=row_group_size)


@pytest.fixture
def parquet_files(tmp_path):
    """Writes two small parquet files of 5 and 4 rows with row groups of 2."""
    first = tmp_path / "a.parquet"
    second = tmp_path / "b.parquet"
    write_parquet(first, ["1920", "1997.0", "c.1904", "None", "1500"], list("ABCDE"), list("vwxyz"), 10, 2)
    write_parquet(second, ["1850", None, "1700", "1999"], list("FGHI"), list("pqrs"), 100, 3)
    return [first, second]


def pixel_value(encoded):
    """Returns the grey brightness of the first pixel of encoded image bytes."""
    with Image.open(BytesIO(encoded)) as image:
        return image.getpixel((0, 0))[0]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1920", 1920),
        ("1997.0", 1997),
        ("c.1904", 1904),
        ("None", None),
        ("", None),
        ("12th century", None),
        ("3000", None),
        (None, None),
        ("1000", 1000),
        ("2029", 2029),
    ],
)
def test_parse_year(text, expected):
    """Years are found in the mixed date forms and absent otherwise."""
    assert parse_year(text) == expected


def test_period_boundaries():
    """The eight periods break at 1599/1600, 1799/1800 and 1949/1950."""
    assert len(PERIODS) == 8
    assert period_of(1000) == 0
    assert period_of(1599) == 0
    assert period_of(1600) == 1
    assert period_of(1799) == 1
    assert period_of(1800) == 2
    assert period_of(1849) == 2
    assert period_of(1850) == 3
    assert period_of(1874) == 3
    assert period_of(1875) == 4
    assert period_of(1899) == 4
    assert period_of(1900) == 5
    assert period_of(1924) == 5
    assert period_of(1925) == 6
    assert period_of(1949) == 6
    assert period_of(1950) == 7
    assert period_of(2029) == 7


def test_read_metadata_numbers_rows_across_files(parquet_files):
    """Years, artists and styles come back in order with global row numbers."""
    metadata = read_metadata(parquet_files)
    assert metadata["row"] == list(range(9))
    assert metadata["year"] == [1920, 1997, 1904, None, 1500, 1850, None, 1700, 1999]
    assert metadata["artist"] == list("ABCDEFGHI")
    assert metadata["style"] == list("vwxyzpqrs")
    assert metadata["title"] == ["t0", "t1", "t2", "t3", "t4", "t0", "t1", "t2", "t3"]


def test_row_group_sizes_lists_the_rows_of_every_group(parquet_files):
    """Each file reports the row count of its row groups, in order."""
    assert row_group_sizes(parquet_files) == [[2, 2, 1], [3, 1]]


def test_read_images_keeps_the_order_given(parquet_files):
    """Unsorted rows across files and row groups return their own images."""
    rows = [8, 0, 5, 4, 2, 7]
    images = read_images(parquet_files, rows)
    assert [pixel_value(image) for image in images] == [103, 10, 100, 14, 12, 102]
    assert len(images) == len(rows)


def test_read_images_handles_repeated_and_empty_requests(parquet_files):
    """A repeated row is returned twice and no rows return nothing."""
    images = read_images(parquet_files, [3, 3])
    assert [pixel_value(image) for image in images] == [13, 13]
    assert read_images(parquet_files, []) == []


def test_read_images_rejects_rows_out_of_range(parquet_files):
    """A row beyond the last file raises IndexError."""
    with pytest.raises(IndexError):
        read_images(parquet_files, [9])


def test_select_balanced_counts_determinism_and_none():
    """At most per_class rows per class, none for None, same seed same pick."""
    labels = [0] * 10 + [1] * 3 + [None] * 5 + [2] * 6
    chosen = select_balanced(labels, 4, seed=1)
    assert chosen == sorted(chosen)
    assert Counter(labels[i] for i in chosen) == {0: 4, 1: 3, 2: 4}
    assert all(labels[i] is not None for i in chosen)
    assert chosen == select_balanced(labels, 4, seed=1)
    assert chosen != select_balanced(labels, 4, seed=2)


def painters_data():
    """Builds a dataset where each painter spans several classes."""
    groups = []
    labels = []
    for painter in range(40):
        for item in range(5):
            groups.append(f"painter{painter}")
            labels.append((painter + item) % 4)
    return groups, labels


def test_split_by_group_has_no_painter_overlap_and_covers_everything():
    """Every index lands in one part and no painter is in two parts."""
    groups, labels = painters_data()
    train, validation, test = split_by_group(groups, labels)
    assert sorted(train + validation + test) == list(range(len(groups)))
    sets = [{groups[i] for i in part} for part in (train, validation, test)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert train == sorted(train)


def test_split_by_group_follows_fractions_and_covers_classes():
    """Part sizes are close to 70/15/15 and every class is in every part."""
    groups, labels = painters_data()
    parts = split_by_group(groups, labels)
    total = len(groups)
    for part, share in zip(parts, (0.7, 0.15, 0.15)):
        assert abs(len(part) / total - share) <= 0.06
        assert set(labels[i] for i in part) == {0, 1, 2, 3}


def test_split_by_group_is_deterministic():
    """The same seed gives the same split."""
    groups, labels = painters_data()
    assert split_by_group(groups, labels, seed=3) == split_by_group(groups, labels, seed=3)


def test_split_by_group_treats_unknown_painters_as_separate_groups():
    """Rows with an unknown painter are spread over parts, not kept together."""
    groups = ["", None, "Unknown Artist", "unknown", "anonymous"] * 20 + ["real"] * 10
    labels = [0] * 110
    train, validation, test = split_by_group(groups, labels)
    assert sorted(train + validation + test) == list(range(110))
    real = {index for index in range(100, 110)}
    holding = [part for part in (train, validation, test) if real & set(part)]
    assert len(holding) == 1
    assert len(validation) > 0 and len(test) > 0
    assert len(train) > 10


class FakeHttpResponse:
    """Stands in for a streamed response carrying JSON or bytes."""

    def __init__(self, payload=b"", listing=None, error=None):
        """Stores the bytes, the JSON body and an optional error to raise."""
        self.payload = payload
        self.listing = listing
        self.error = error

    def __enter__(self):
        """Enters the context and returns the response itself."""
        return self

    def __exit__(self, *details):
        """Leaves the context without swallowing errors."""
        return False

    def raise_for_status(self):
        """Raises the stored error, if any."""
        if self.error:
            raise self.error

    def json(self):
        """Returns the stored JSON body."""
        return self.listing

    def iter_content(self, chunk_size):
        """Serves the whole payload as one chunk."""
        yield self.payload


def fake_get_factory(calls, fail_on=None):
    """Builds a fake ``get`` that records urls and serves a two-file listing."""
    listing = {
        "siblings": [
            {"rfilename": "README.md"},
            {"rfilename": "data/train-00001-of-00002.parquet"},
            {"rfilename": "data/train-00000-of-00002.parquet"},
        ]
    }

    def fake_get(url, **kwargs):
        """Serves the listing for the API url and file bytes for the others."""
        calls.append(url)
        if url == paintings.WIKIART_API_URL:
            return FakeHttpResponse(listing=listing)
        if fail_on and url.endswith(fail_on):
            return FakeHttpResponse(error=HTTPError("500"))
        return FakeHttpResponse(payload=url.encode())

    return fake_get


def test_download_wikiart_fetches_each_file_once(tmp_path, monkeypatch):
    """Files are saved, returned sorted, and a second call fetches nothing."""
    calls = []
    monkeypatch.setattr(paintings, "get", fake_get_factory(calls))
    paths = download_wikiart(tmp_path / "wiki")
    assert [path.name for path in paths] == [
        "train-00000-of-00002.parquet",
        "train-00001-of-00002.parquet",
    ]
    assert paths[0].read_bytes().endswith(b"train-00000-of-00002.parquet")
    assert len(calls) == 3
    assert (tmp_path / "wiki" / ".complete").exists()
    assert download_wikiart(tmp_path / "wiki") == paths
    assert len(calls) == 3


def test_download_wikiart_failure_leaves_no_marker(tmp_path, monkeypatch):
    """A failed file download raises, writes no marker and keeps no partial file."""
    monkeypatch.setattr(paintings, "get", fake_get_factory([], fail_on="00001-of-00002.parquet"))
    with pytest.raises(HTTPError):
        download_wikiart(tmp_path / "wiki")
    assert not (tmp_path / "wiki" / ".complete").exists()
    assert not (tmp_path / "wiki" / "train-00001-of-00002.parquet").exists()


def test_download_wikiart_resumes_after_failure(tmp_path, monkeypatch):
    """A retry keeps files already finished and fetches only the rest."""
    monkeypatch.setattr(paintings, "get", fake_get_factory([], fail_on="00001-of-00002.parquet"))
    with pytest.raises(HTTPError):
        download_wikiart(tmp_path / "wiki")
    calls = []
    monkeypatch.setattr(paintings, "get", fake_get_factory(calls))
    paths = download_wikiart(tmp_path / "wiki")
    assert len(paths) == 2
    assert not any(url.endswith("00000-of-00002.parquet") for url in calls)


def test_decode_images_shape_and_dtype():
    """Decoded images form a uint8 batch of the requested square size."""
    batch = decode_images([encoded_image(50, 16), encoded_image(200, 32)], 8)
    assert batch.shape == (2, 3, 8, 8)
    assert batch.dtype == torch.uint8
    assert int(batch[0, 0, 0, 0]) == 50
    assert int(batch[1, 2, 7, 7]) == 200
    assert decode_images([], 8).shape == (0, 3, 8, 8)


def test_encoded_image_dataset_decodes_on_access():
    """Items are transformed tensors paired with their labels."""
    dataset = EncodedImageDataset([encoded_image(255), encoded_image(0)], [3, 5], ToTensor())
    assert len(dataset) == 2
    image, label = dataset[0]
    assert image.shape == (3, 8, 8)
    assert float(image.max()) == pytest.approx(1.0)
    assert label == 3
    assert dataset[1][1] == 5


def test_accuracy_by_group_matches_hand_computed_values():
    """Accuracy and mean period distance are computed per group and sorted by count."""
    groups = ["a"] * 4 + ["b"] * 2 + ["c"] * 1
    targets = [0, 1, 2, 3, 4, 5, 0]
    predictions = [0, 1, 4, 0, 4, 7, 0]
    result = accuracy_by_group(targets, predictions, groups, minimum=2)
    assert [entry["group"] for entry in result] == ["a", "b"]
    assert result[0] == {"group": "a", "count": 4, "accuracy": 0.5, "mean_distance": 1.25}
    assert result[1] == {"group": "b", "count": 2, "accuracy": 0.5, "mean_distance": 1.0}


def test_accuracy_by_group_applies_the_minimum():
    """Groups below the minimum are left out, and the default minimum is 20."""
    groups = ["a"] * 20 + ["b"] * 19
    targets = [0] * 39
    predictions = [0] * 39
    assert [entry["group"] for entry in accuracy_by_group(targets, predictions, groups)] == ["a"]
    assert accuracy_by_group(targets, predictions, groups, minimum=40) == []


def test_select_examples_takes_hits_then_misses_from_different_classes():
    """Three hits and three misses come from six different true classes when possible."""
    targets = [label for label in range(8) for _ in range(4)]
    predictions = [label if position % 2 == 0 else (label + 1) % 8 for position, label in enumerate(targets)]
    chosen = select_examples(targets, predictions, per_outcome=3)
    hits, misses = chosen[:3], chosen[3:]
    assert len(chosen) == 6
    assert all(targets[index] == predictions[index] for index in hits)
    assert all(targets[index] != predictions[index] for index in misses)
    assert len({targets[index] for index in chosen}) == 6
    assert [targets[index] for index in hits] == [0, 4, 7]


def test_select_examples_takes_the_first_sample_of_each_chosen_class():
    """Within a class, the sample with the lowest index is the one chosen."""
    targets = [0, 0, 1, 1, 2, 2]
    predictions = [0, 0, 0, 0, 2, 2]
    assert select_examples(targets, predictions, per_outcome=1) == [0, 2]


def test_select_examples_is_deterministic():
    """The same inputs always give the same choice."""
    targets = [3, 1, 2, 0, 3, 1, 2, 0]
    predictions = [3, 0, 2, 1, 0, 1, 0, 0]
    assert select_examples(targets, predictions, 2) == select_examples(targets, predictions, 2)


def test_select_examples_reuses_classes_and_returns_fewer_when_short():
    """With few classes or few hits, classes repeat and the list may be shorter."""
    targets = [0, 0, 0, 1, 1]
    predictions = [0, 0, 1, 0, 0]
    chosen = select_examples(targets, predictions, per_outcome=3)
    assert chosen == [0, 1, 2, 3, 4]
    assert select_examples([0, 1], [0, 1], per_outcome=3) == [0, 1]
    assert select_examples([], [], per_outcome=3) == []


def test_spelling_variants_groups_names_that_differ_only_in_form():
    """Case, accents, punctuation, spacing and word order do not make a new painter."""
    names = [
        "Claude Monet",
        "claude monet",
        "Monet, Claude",
        "Paul Cézanne",
        "Paul Cezanne",
        "Edgar Degas",
        "claude-monet",
        None,
        "Claude Monet",
    ]
    assert spelling_variants(names) == [
        ["Claude Monet", "Monet, Claude", "claude monet", "claude-monet"],
        ["Paul Cezanne", "Paul Cézanne"],
    ]


def test_spelling_variants_is_empty_when_every_name_has_one_form():
    """Distinct painters, repeated or not, give no variant group."""
    assert spelling_variants(["A B", "A B", "C D", "", None]) == []
