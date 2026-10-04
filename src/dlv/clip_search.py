"""Zero-shot concept ranking and text-to-image search with a pre-trained CLIP model."""

from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.nn.functional import normalize
from transformers import CLIPModel, CLIPProcessor

from dlv.data import file_digest, is_readable_image

CLIP_CHECKPOINT = "openai/clip-vit-base-patch32"


def list_ads16_images(root: Path) -> list[dict]:
    """Lists the distinct readable images of the ADS-16 archive.

    Files under an ``Ads`` folder are advertisements and carry the integer
    name of their numbered category folder; files under a ``Corpus`` folder
    belong to the participants and have no category. Other files, unreadable
    files and exact copies by content are left out. Of a group of copies the
    first path in sorted order is kept. The result is sorted by path.
    """
    root = Path(root)
    entries = []
    seen = set()
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        parts = path.relative_to(root).parts[:-1]
        if "Ads" in parts:
            source = "anuncio"
        elif "Corpus" in parts:
            source = "participante"
        else:
            continue
        if not is_readable_image(path):
            continue
        digest = file_digest(path)
        if digest in seen:
            continue
        seen.add(digest)
        category = int(path.parent.name) if source == "anuncio" and path.parent.name.isdigit() else None
        entries.append({"path": path, "source": source, "category": category})
    return entries


def load_clip(device) -> tuple:
    """Loads the pre-trained CLIP ViT-B/32 and its processor, in evaluation mode on ``device``."""
    model = CLIPModel.from_pretrained(CLIP_CHECKPOINT).to(device).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_CHECKPOINT)
    return model, processor


def projected_features(output) -> Tensor:
    """Returns the projected embeddings from a CLIP feature call, whatever the library version returns."""
    return output if isinstance(output, Tensor) else output.pooler_output


@torch.no_grad()
def embed_images(model, processor, paths: list[Path], device, batch_size: int = 64) -> Tensor:
    """Embeds images with CLIP into unit-length float32 rows on the CPU.

    Every file is opened as RGB and processed in batches of ``batch_size``.
    """
    batches = []
    for start in range(0, len(paths), batch_size):
        images = []
        for path in paths[start : start + batch_size]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        features = projected_features(model.get_image_features(pixel_values=pixel_values))
        batches.append(normalize(features.float(), dim=-1).cpu())
    if not batches:
        return torch.empty(0, 0)
    return torch.cat(batches)


@torch.no_grad()
def embed_texts(model, processor, texts: list[str], device) -> Tensor:
    """Embeds texts with CLIP into unit-length float32 rows on the CPU.

    Texts longer than the tokenizer's limit are truncated, and a warning
    naming each of them is printed.
    """
    limit = processor.tokenizer.model_max_length
    lengths = processor(text=texts, padding=False, truncation=False)["input_ids"]
    for text, ids in zip(texts, lengths):
        if len(ids) > limit:
            print(f"Warning: text truncated to {limit} tokens ({len(ids)} needed): {text!r}")
    encoded = processor(text=texts, padding=True, truncation=True, return_tensors="pt")
    features = projected_features(
        model.get_text_features(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        )
    )
    return normalize(features.float(), dim=-1).cpu()


def cached(path: Path, compute) -> Tensor:
    """Loads the tensor stored at ``path``, or calls ``compute()``, saves its result there and returns it."""
    path = Path(path)
    if path.exists():
        return torch.load(path)
    result = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    return result


def similarity_matrix(image_embeddings: Tensor, text_embeddings: Tensor) -> Tensor:
    """Returns the cosine similarity of every image to every text, shaped (images, texts)."""
    return normalize(image_embeddings, dim=-1) @ normalize(text_embeddings, dim=-1).T


def concept_probabilities(similarity: Tensor, logit_scale: float = 100.0) -> Tensor:
    """Applies CLIP's zero-shot rule: a softmax over the concepts of each image's scaled similarities."""
    return (similarity * logit_scale).softmax(dim=-1)


def rank_concepts(
    similarity: Tensor, concepts: list[str], threshold: float, logit_scale: float = 100.0
) -> list[dict]:
    """Ranks concepts by how often they are the likely content of an image.

    For each concept the result gives its mean similarity, its mean
    probability, the share of images whose probability for it is at or above
    ``threshold`` (``frequency``) and the number of such images (``count``).
    The list is sorted by frequency, then by mean similarity, largest first.
    """
    probabilities = concept_probabilities(similarity, logit_scale)
    present = probabilities >= threshold
    ranking = [
        {
            "concept": concept,
            "mean_similarity": similarity[:, index].mean().item(),
            "mean_probability": probabilities[:, index].mean().item(),
            "frequency": present[:, index].float().mean().item(),
            "count": int(present[:, index].sum().item()),
        }
        for index, concept in enumerate(concepts)
    ]
    return sorted(ranking, key=lambda row: (row["frequency"], row["mean_similarity"]), reverse=True)


def top_images(similarity: Tensor, concept_index: int, k: int) -> list[int]:
    """Returns the indices of the ``k`` images most similar to one concept, best first."""
    scores = similarity[:, concept_index]
    return scores.topk(min(k, scores.shape[0])).indices.tolist()


def search(query_embedding: Tensor, image_embeddings: Tensor, k: int = 5) -> list[tuple[int, float]]:
    """Finds the ``k`` images most similar to a query embedding.

    Returns ``(index, score)`` pairs, best first, with the cosine similarity
    as the score. Fewer pairs come back when the corpus has fewer images.
    """
    scores = similarity_matrix(image_embeddings, query_embedding.reshape(1, -1))[:, 0]
    best = scores.topk(min(k, scores.shape[0]))
    return list(zip(best.indices.tolist(), best.values.tolist()))


def describe_tokens(tokenizer, texts: list[str]) -> list[dict]:
    """Shows how a tokenizer splits texts, padded to the longest of them.

    Per text: the ``tokens``, the ``input_ids``, the ``attention_mask`` and
    ``padding``, the number of padding positions. Works with any Hugging Face
    tokenizer, such as CLIP's or BERT's.
    """
    encoded = tokenizer(texts, padding=True)
    return [
        {
            "tokens": tokenizer.convert_ids_to_tokens(ids),
            "input_ids": list(ids),
            "attention_mask": list(mask),
            "padding": list(mask).count(0),
        }
        for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"])
    ]
