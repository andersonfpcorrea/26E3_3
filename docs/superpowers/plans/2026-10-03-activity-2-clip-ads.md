# Activity 2: Semantic Search in Advertising with CLIP - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Deliver the Activity 2 notebook and report section: rank concepts by how often they appear in the ADS-16 images and search images by text, using pre-trained CLIP with no training.

**Architecture:** New module `clip_search.py` in `src/dlv`, tested on CPU with stand-in embeddings. Notebook built from `notebooks_src/a2/`, run on a Colab T4.

**Tech Stack:** Python 3.13, PyTorch 2.11, Hugging Face `transformers` (CLIP `openai/clip-vit-base-patch32`), matplotlib, pytest, Colab CLI.

**Spec:** `docs/superpowers/specs/2026-10-03-computer-vision-project-design.md`

This is plan 4 of 5. It states interfaces, tests and acceptance checks; the implementer writes the code. Global constraints are the ones in `2026-10-03-activity-1-vision-transformers.md`, section "Global Constraints", and bind this plan too.

## Facts already verified

- Dataset: Kaggle `groffo/ads16-dataset`, 0.8 GB, no login, through `download_kaggle_dataset`.
- Layout: `ADS16_Benchmark_part1/ADS16_Benchmark_part1/Ads/Ads/<category>/<n>.png` and the same under `part2`: 300 advertisement images in numbered category folders. `.../Corpus/Corpus/<user>/...`: images supplied by the study's participants, plus CSV files. In total 2,696 PNG and 1 JPG files, 720 CSV, 6 nested ZIP files, a few text and PDF documents.
- The assignment asks for the ADS-16 image corpus or at least 500 images. The 300 advertisements alone are below 500, so the corpus is the advertisements plus the participants' images, and the notebook reports results for both parts.
- The assignment speaks of 16 product categories; the archive has numbered category folders. Report the number actually found.

## Review Focus

1. A corrupt or non-image file in the archive: skipped at listing, never a crash mid-embedding.
2. The same image stored twice: exact copies removed before ranking, so frequency is not inflated.
3. A text query longer than CLIP's 77-token limit: truncated with a printed warning, not an error.
4. A search asks for more results than there are images: returns what exists.
5. Embeddings are recomputed on every run of a cell: they are cached on disk and reused.

---

### Task 1: CLIP search module

**Files:** `src/dlv/clip_search.py`, `tests/test_clip_search.py`, `pyproject.toml` and `uv.lock` (add `transformers`).

**Produces:**

- `list_ads16_images(root: Path) -> list[dict]`: one entry per distinct readable image: `path`, `source` (`"anuncio"` for files under an `Ads` folder, `"participante"` for files under `Corpus`), `category` (the numbered folder as int for advertisements, `None` otherwise). Exact copies by content removed. Sorted by path. Reuse helpers from `dlv.data` through a single-statement import.
- `load_clip(device) -> tuple[model, processor]`: `openai/clip-vit-base-patch32`, evaluation mode.
- `embed_images(model, processor, paths: list[Path], device, batch_size: int = 64) -> Tensor`: unit-length image embeddings `(n, d)` on CPU, float32.
- `embed_texts(model, processor, texts: list[str], device) -> Tensor`: unit-length text embeddings; texts over the model's token limit are truncated and a warning is printed naming them.
- `cached(path: Path, compute) -> Tensor`: loads the tensor at `path` when it exists, otherwise calls `compute()`, saves and returns it.
- `similarity_matrix(image_embeddings: Tensor, text_embeddings: Tensor) -> Tensor`: cosine similarities `(images, texts)`.
- `concept_probabilities(similarity: Tensor, logit_scale: float = 100.0) -> Tensor`: softmax over the concepts for each image, CLIP's zero-shot rule.
- `rank_concepts(similarity: Tensor, concepts: list[str], threshold: float, logit_scale: float = 100.0) -> list[dict]`: per concept `concept`, `mean_similarity`, `mean_probability`, `frequency` (share of images whose probability for the concept is at or above the threshold), `count`; sorted by frequency then mean similarity, largest first.
- `top_images(similarity: Tensor, concept_index: int, k: int) -> list[int]`: image indices with the highest similarity to one concept.
- `search(query_embedding: Tensor, image_embeddings: Tensor, k: int = 5) -> list[tuple[int, float]]`: the `k` most similar images with scores, fewer when there are fewer images.
- `describe_tokens(tokenizer, texts: list[str]) -> list[dict]`: per text the tokens, `input_ids`, `attention_mask` and the number of padding positions, for a tokenizer called with padding to the longest text. Works for both a CLIP tokenizer and a BERT tokenizer.

**Tests, at least**, with stand-in objects and tensors, never downloading a model: listing by source and category, copies removed, unreadable files skipped; `cached` computes once; similarity of unit vectors against hand-computed values; probabilities sum to one per image; ranking order, frequency and count against a hand-built similarity matrix, and the effect of the threshold; `top_images` order; `search` order, scores, and `k` larger than the corpus; `embed_images` and `embed_texts` return unit-length rows when given a fake model and processor; `describe_tokens` with a fake tokenizer.

Commit message: `Add CLIP search module`.

### Task 2: Activity 2 notebook and Colab run

**Files:** `notebooks_src/a2/*`, `scripts/build_notebooks.py` (`NOTEBOOKS` gains `"a2": "A2_clip_ads16"`), `src/dlv/figures.py` and its test (new plots), run outputs `notebooks/A2_clip_ads16.ipynb`, `results/a2.json`, `report/figures/a2_*.png`.

**Data:** download; list; print images found, copies removed, counts by source and by advertisement category. Embed every image once, cached.

**2.1 Ranking:** at least 24 concepts as English prompts in the form "a photo of ...", covering objects, people, scenes and advertising elements (for example a car, a person, a smiling woman, food, a drink, a mobile phone, a computer, clothes, shoes, jewellery, cosmetics, furniture, a house, outdoor scenery, a beach, an animal, a child, sports, text and logo, a product on a plain background). Compute the similarity matrix. Choose the presence threshold from the data: plot the distribution of the top probability per image and of all probabilities, inspect images just above and just below candidate thresholds, and state the rule and why. Report the ranking with mean similarity and frequency, for the whole corpus and separately for advertisements and for participants' images. Show the five most frequent concepts with six example images each.

**2.2 Search:** at least ten text queries with the top five images each, from generic to specific and from concrete to abstract (for example: "a car"; "a red sports car on a road"; "a happy family at home"; "luxury"; "freedom"; "something healthy to eat"; "technology that makes life easier"; "an advertisement with a lot of text"; "loneliness"; one query in Portuguese). Save for each query the indices, scores and sources of the results.

**Tokenisation:** show `describe_tokens` for three queries of different length with the CLIP tokenizer and with a BERT tokenizer (`bert-base-uncased`), so the report can compare padding and attention mask.

**Metrics file keys:** `files_found`, `duplicates_removed`, `images`, `by_source`, `ad_categories`, `concepts`, `threshold` (value and rule), `ranking` (whole corpus, advertisements, participants), `top_examples`, `queries` (text, results), `tokenisation`, `embedding_seconds`.

**Figures:** probability distributions with the threshold marked; ranking bars; one grid per top-five concept; one grid per query with the query as title and the score under each image.

**Process:** as in Activity 1 Task 5: local checks first; `make colab-a2` in the background with output redirected under `build/`; budget 4 Colab runs; under 30 minutes; header values from the final run; `make check-notebooks` must not name this notebook. Look at every grid with the Read tool and record, for each query, whether the five images match the query and what is unexpected. Describe only what you see.

Commit messages: one per addition to `src/dlv`, then `Add Activity 2 notebook with Colab T4 run`.

### Task 3: Activity 2 report section

**Files:** `report/secoes/20_atividade2.md`, an analysis cell in `notebooks_src/a2/` before the last cell.

Headings in this order:

- `# Atividade 2: Reconhecimento semântico em publicidade com CLIP`
- `## Problema`: the corpus as found, its two parts, counts.
- `## Como o CLIP alinha imagem e texto` (rubric: alignment of visual and textual representations; why contrastive pre-training enables retrieval with no supervised training).
- `## Ranking de conceitos` (rubric: at least 20 descriptions, top five with examples): the prompts, the threshold rule with its justification, the ranking, the grids, what the ranking says about the corpus, and where the threshold rule is weak.
- `## Busca por texto` (rubric: at least eight queries varying in specificity, documented and analysed): one subsection per query with the grid and a short analysis: what the model retrieved, whether it matches, what was unexpected. Then a summary by type of query.
- `## Consulta de texto no CLIP e tokenização no BERT` (rubric: comparison, role of padding and attention mask): what each tokenizer does with the same texts, what padding is for, what the attention mask is for, how each model turns the sequence into one vector (CLIP: the end-of-text token position; BERT: the CLS token).
- `## Análise crítica`: bias of the model, English prompts, sensitivity to prompt wording, images that are not advertisements, what would be changed.

Then rerun the notebook if a cell changed, `make report`, read the PDF pages, `make zip`. Budget: 2 Colab runs.

Commit message: `Add Activity 2 report section`.
