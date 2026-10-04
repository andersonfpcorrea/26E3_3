"""Tests for the CLIP search module, using stand-in models and tiny image trees."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from dlv.clip_search import (
    cached,
    concept_probabilities,
    describe_tokens,
    embed_images,
    embed_texts,
    list_ads16_images,
    rank_concepts,
    search,
    similarity_matrix,
    top_images,
)


def make_archive(root):
    """Builds a small ADS-16 like tree with copies, a corrupt file and stray files."""
    ads = root / "part1" / "Ads" / "Ads"
    corpus = root / "part1" / "Corpus" / "Corpus" / "user1"
    for folder in (ads / "1", ads / "10", corpus):
        folder.mkdir(parents=True)
    Image.new("RGB", (8, 8), (10, 0, 0)).save(ads / "1" / "1.png")
    Image.new("L", (9, 9), 50).save(ads / "1" / "2.png")
    Image.new("RGB", (8, 8), (10, 0, 0)).save(ads / "10" / "3.png")
    Image.new("RGB", (8, 8), (0, 90, 0)).save(corpus / "a.png")
    Image.new("RGB", (8, 8), (0, 0, 200)).save(corpus / "b.jpg")
    (corpus / "broken.png").write_bytes(b"not an image")
    (corpus / "table.csv").write_text("a,b")
    Image.new("RGB", (8, 8), (7, 7, 7)).save(root / "part1" / "stray.png")
    return root


def test_list_ads16_images_sources_categories_and_order(tmp_path):
    """Entries carry source and category, skip stray files and come sorted by path."""
    entries = list_ads16_images(make_archive(tmp_path))
    assert [entry["path"] for entry in entries] == sorted(entry["path"] for entry in entries)
    by_name = {entry["path"].name: entry for entry in entries}
    assert by_name["2.png"]["source"] == "anuncio"
    assert by_name["2.png"]["category"] == 1
    assert by_name["a.png"]["source"] == "participante"
    assert by_name["a.png"]["category"] is None
    assert "stray.png" not in by_name
    assert "table.csv" not in by_name


def test_list_ads16_images_removes_copies_and_unreadable_files(tmp_path):
    """Exact copies keep their first path and corrupt files are skipped."""
    entries = list_ads16_images(make_archive(tmp_path))
    names = [entry["path"].name for entry in entries]
    assert "3.png" not in names
    assert "1.png" in names
    assert "broken.png" not in names
    assert len(entries) == 4


class FakeClip:
    """Maps every image or text to a fixed non-unit vector derived from its input."""

    def get_image_features(self, pixel_values):
        """Returns the first three values of each image, scaled up."""
        return pixel_values[:, 0, 0, :3] * 5 + 1

    def get_text_features(self, input_ids, attention_mask):
        """Returns the token ids as vectors, ignoring the padding."""
        return (input_ids * attention_mask).float()[:, :3] + 1


class FakeOutput:
    """Mimics the output object newer library versions return."""

    def __init__(self, pooler_output):
        """Stores the projected embeddings."""
        self.pooler_output = pooler_output


class FakeClipWithOutputs(FakeClip):
    """Wraps every feature call in an output object."""

    def get_image_features(self, pixel_values):
        """Wraps the image embeddings."""
        return FakeOutput(super().get_image_features(pixel_values))

    def get_text_features(self, input_ids, attention_mask):
        """Wraps the text embeddings."""
        return FakeOutput(super().get_text_features(input_ids, attention_mask))


class FakeTokenizer:
    """Splits on spaces, maps words to ids and pads to the longest text."""

    model_max_length = 4
    vocabulary = {"<pad>": 0, "a": 1, "red": 2, "car": 3, "on": 4, "the": 5, "road": 6}

    def __call__(self, texts, padding=False, truncation=False, return_tensors=None):
        """Encodes texts as a dict of id and mask lists, or tensors."""
        ids = [[self.vocabulary[word] for word in text.split()] for text in texts]
        if truncation:
            ids = [row[: self.model_max_length] for row in ids]
        masks = [[1] * len(row) for row in ids]
        if padding:
            width = max(len(row) for row in ids)
            masks = [mask + [0] * (width - len(mask)) for mask in masks]
            ids = [row + [0] * (width - len(row)) for row in ids]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}
        return {"input_ids": ids, "attention_mask": masks}

    def convert_ids_to_tokens(self, ids):
        """Maps ids back to words."""
        names = {value: key for key, value in self.vocabulary.items()}
        return [names[i] for i in ids]


class FakeProcessor:
    """Processes images into tensors and texts through the fake tokenizer."""

    def __init__(self):
        """Creates the tokenizer and an empty record of image batch sizes."""
        self.tokenizer = FakeTokenizer()
        self.batch_sizes = []

    def __call__(self, images=None, text=None, **kwargs):
        """Returns pixel values for images, or token tensors for text."""
        if images is not None:
            self.batch_sizes.append(len(images))
            assert all(image.mode == "RGB" for image in images)
            pixels = torch.stack(
                [torch.tensor(image.getpixel((0, 0)), dtype=torch.float32)[None, None, :] / 255 for image in images]
            )
            return {"pixel_values": pixels.permute(0, 2, 1, 3).expand(-1, 3, 2, 3)}
        return self.tokenizer(text, **kwargs)


def make_images(tmp_path, count=5):
    """Writes small images with distinct colours and returns their paths, mixing modes."""
    paths = []
    for index in range(count):
        path = tmp_path / f"{index}.png"
        mode = "L" if index % 2 else "RGB"
        Image.new(mode, (6 + index, 6), 40 * index + 10).save(path)
        paths.append(path)
    return paths


@pytest.mark.parametrize("model", [FakeClip(), FakeClipWithOutputs()])
def test_embed_images_returns_unit_rows_in_batches(tmp_path, model):
    """Image embeddings are unit-length float32 CPU rows, built in batches."""
    processor = FakeProcessor()
    embeddings = embed_images(model, processor, make_images(tmp_path), "cpu", batch_size=2)
    assert embeddings.shape == (5, 3)
    assert embeddings.dtype == torch.float32
    assert embeddings.device.type == "cpu"
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(5), atol=1e-6)
    assert processor.batch_sizes == [2, 2, 1]


@pytest.mark.parametrize("model", [FakeClip(), FakeClipWithOutputs()])
def test_embed_texts_returns_unit_rows(model):
    """Text embeddings are unit-length rows, one per text."""
    embeddings = embed_texts(model, FakeProcessor(), ["a car", "a red car"], "cpu")
    assert embeddings.shape == (2, 3)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(2), atol=1e-6)


def test_embed_texts_warns_about_truncated_texts(capsys):
    """A text over the token limit is truncated, named in a warning, and does not raise."""
    long_text = "a red car on the road"
    embeddings = embed_texts(FakeClip(), FakeProcessor(), ["a car", long_text], "cpu")
    printed = capsys.readouterr().out
    assert long_text in printed
    assert "a car'" not in printed
    assert embeddings.shape[0] == 2


def test_embed_texts_is_silent_within_the_limit(capsys):
    """No warning is printed when every text fits."""
    embed_texts(FakeClip(), FakeProcessor(), ["a car", "a red car"], "cpu")
    assert capsys.readouterr().out == ""


def test_cached_computes_once(tmp_path):
    """The second call loads the saved tensor and does not compute again."""
    calls = []

    def compute():
        """Counts the calls and returns a tensor."""
        calls.append(1)
        return torch.arange(4.0)

    path = tmp_path / "nested" / "embeddings.pt"
    first = cached(path, compute)
    second = cached(path, compute)
    assert len(calls) == 1
    assert torch.equal(first, second)
    assert path.exists()


def test_similarity_matrix_of_unit_vectors():
    """Similarities match hand-computed dot products."""
    images = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
    texts = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]])
    expected = torch.tensor([[1.0, 0.0, 0.8], [0.6, 0.8, 0.96]])
    assert torch.allclose(similarity_matrix(images, texts), expected, atol=1e-6)


def test_concept_probabilities_sum_to_one_per_image():
    """Each row of probabilities sums to one and favours the larger similarity."""
    similarity = torch.tensor([[0.3, 0.2, 0.1], [0.1, 0.1, 0.4]])
    probabilities = concept_probabilities(similarity)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2), atol=1e-6)
    assert probabilities[0].argmax().item() == 0
    assert probabilities[1].argmax().item() == 2


SIMILARITY = torch.tensor(
    [
        [0.30, 0.20, 0.10],
        [0.30, 0.25, 0.10],
        [0.20, 0.30, 0.10],
        [0.10, 0.20, 0.30],
    ]
)


def test_rank_concepts_order_frequency_and_count():
    """With a sharp scale each image votes for its best concept."""
    ranking = rank_concepts(SIMILARITY, ["a", "b", "c"], threshold=0.5, logit_scale=1000.0)
    assert [row["concept"] for row in ranking] == ["a", "b", "c"]
    assert [row["count"] for row in ranking] == [2, 1, 1]
    assert [row["frequency"] for row in ranking] == [0.5, 0.25, 0.25]
    assert ranking[0]["mean_similarity"] == pytest.approx(0.225)
    assert ranking[0]["mean_probability"] == pytest.approx(0.5, abs=1e-3)


def test_rank_concepts_breaks_frequency_ties_by_mean_similarity():
    """Equal frequency is ordered by mean similarity, largest first."""
    ranking = rank_concepts(SIMILARITY, ["a", "b", "c"], threshold=0.5, logit_scale=1000.0)
    assert ranking[1]["concept"] == "b"
    assert ranking[1]["mean_similarity"] > ranking[2]["mean_similarity"]


def test_rank_concepts_threshold_changes_frequency():
    """A threshold of zero counts every image for every concept; a high one counts none."""
    everything = rank_concepts(SIMILARITY, ["a", "b", "c"], threshold=0.0)
    assert [row["count"] for row in everything] == [4, 4, 4]
    nothing = rank_concepts(SIMILARITY, ["a", "b", "c"], threshold=1.01)
    assert [row["frequency"] for row in nothing] == [0.0, 0.0, 0.0]


def test_top_images_order_and_limit():
    """Indices come best first, and k above the corpus returns every image."""
    assert top_images(SIMILARITY, 0, 2) == [0, 1]
    assert top_images(SIMILARITY, 2, 1) == [3]
    assert sorted(top_images(SIMILARITY, 1, 10)) == [0, 1, 2, 3]


def test_search_order_and_scores():
    """Results are ordered by cosine similarity and carry the score."""
    images = torch.tensor([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])
    results = search(torch.tensor([1.0, 0.0]), images, k=2)
    assert [index for index, _ in results] == [0, 1]
    assert results[0][1] == pytest.approx(1.0)
    assert results[1][1] == pytest.approx(0.6)


def test_search_with_k_larger_than_corpus_returns_everything():
    """Asking for more results than images returns every image."""
    images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    results = search(torch.tensor([0.0, 1.0]), images, k=10)
    assert [index for index, _ in results] == [1, 0]


def test_describe_tokens_with_fake_tokenizer():
    """Tokens, ids, masks and padding counts follow padding to the longest text."""
    description = describe_tokens(FakeTokenizer(), ["a car", "a red car on"])
    assert description[0]["tokens"] == ["a", "car", "<pad>", "<pad>"]
    assert description[0]["input_ids"] == [1, 3, 0, 0]
    assert description[0]["attention_mask"] == [1, 1, 0, 0]
    assert description[0]["padding"] == 2
    assert description[1]["padding"] == 0
    assert description[1]["tokens"] == ["a", "red", "car", "on"]
