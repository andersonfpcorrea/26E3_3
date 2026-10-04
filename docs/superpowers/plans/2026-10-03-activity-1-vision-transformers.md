# Activity 1: Dating Paintings with Vision Transformers - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Deliver the Activity 1 notebook and report section: a Vision Transformer built by hand and trained from scratch, a pre-trained ViT fine-tuned on the same data, both dating paintings into eight time periods.

**Architecture:** New modules `attention.py`, `vit.py` and `paintings.py` in `src/dlv`, tested on CPU. The notebook is built from `notebooks_src/a1/` by the existing builder and run on a Colab T4 by the existing runner.

**Tech Stack:** Python 3.13, PyTorch 2.11, torchvision 0.26, pyarrow, scikit-learn, matplotlib, pytest, Colab CLI.

**Spec:** `docs/superpowers/specs/2026-10-03-computer-vision-project-design.md`

This is plan 2 of 5. Plan 1 (`2026-10-03-foundation-and-activity-3.md`) is merged. This plan fixes interfaces, tests and acceptance checks; the implementer writes the code.

## Global Constraints

- Code, identifiers, docstrings and commit messages in English. Notebook prose, figure labels and report in Portuguese.
- No `#` comments. A docstring on every function, method and class, tests included.
- Prefer named imports. Plain `import module` only for `sys`, `re`, `json`, `time`, `random`, `torch`, `nbformat`, `numpy as np`, `matplotlib.pyplot as plt`, and a script module a test must monkeypatch.
- No emojis.
- Commit messages: one line, no trailer of any kind, no mention of AI tools. Check with `git log -1 --format=%B`.
- Stage only the files you changed. Never `git add -A`.
- `src/dlv` is the only place where logic lives. Cell files hold configuration, calls and prints.
- Each notebook is self-contained, runs start to finish on a Colab T4 with no edits, and stays under 30 minutes.
- Cell files are plain Python. No `!` or `%` lines. Every name a cell uses from the package comes through a single-statement `from dlv.<module> import <names>` at top level, and that module is listed in a `.modules` file earlier in the folder.
- Inlined modules share one namespace: no two modules may define the same top-level name.
- Tests run on CPU with tiny inputs and never download data or weights. Run one test file while iterating. Run the whole suite once per task, after `uptime`.
- Colab: never leave a session running. After every run, `colab sessions`, and stop any session named `dlv-<activity>`. Do not pipe `make colab-*` output through anything that closes the pipe early. Budget per task is stated in the task.
- Report prose: simple Portuguese, short sentences, bullet points. No "não é A, é B" construction. No dashes as punctuation. None of: robusto, crucial, fundamental, abrangente, poderoso, notável, vale ressaltar, é importante destacar. Brazilian number format. Every number comes from the results file of the final run.
- Results are reported as they are. Do not tune hyperparameters to reach a target score. A weak result is written up honestly.

## Facts already verified

- Dataset: Hugging Face `Artificio/WikiArt`. 103,250 rows, images already 256x256, four parquet files, 1.7 GB, no login. File list: `https://huggingface.co/api/datasets/Artificio/WikiArt` (field `siblings`, entries under `data/`). Download: `https://huggingface.co/datasets/Artificio/WikiArt/resolve/main/<path>`.
- Columns: `title`, `artist`, `date`, `genre`, `style`, `description`, `filename`, `image` (struct with `bytes`), `embeddings_pca512`. Read only the columns needed.
- `date` is a string in mixed forms: `1920`, `1997.0`, `c.1904`, `None`. About 78% hold a year.
- Estimated paintings per period in the full set: before 1600 about 5,800; 1600 to 1799 about 7,100; 1800 to 1849 about 5,000; 1850 to 1874 about 4,800; 1875 to 1899 about 13,000; 1900 to 1924 about 16,800; 1925 to 1949 about 10,900; 1950 onward about 16,600.
- Colab stops a notebook cell after 30 seconds unless the runner passes `--timeout`; the runner passes 1800.
- A Colab run of the Activity 3 notebook takes about 4 minutes end to end.

## Review Focus

1. A painter appears in both training and test: the split must make that impossible, and a test must prove it on data where one painter has paintings in several periods.
2. A `date` string holds no year, or a year outside 1000 to 2029: the row is left out, never guessed.
3. The attention function receives a mask that hides every position of a row: no NaN in the output.
4. The ViT is called with an image whose size is not the configured one: a clear error naming both sizes.
5. The notebook is run twice in one session: nothing is downloaded twice.

---

### Task 1: Foundation carry-overs

**Files:** `scripts/build_notebooks.py`, `src/dlv/engine.py`, `Makefile`, `notebooks_src/a3/*`, their tests.

- **Builder false collision.** Top-level definitions are the names bound by `def`, `class`, `import`, and the targets of assignments (including tuple targets, annotated and augmented assignments, and walrus expressions). Loop variables of comprehensions and the name `__all__` are not definitions. Tests: two modules each with a module-level list comprehension using the same loop variable build fine; two modules each setting `__all__` build fine; two modules assigning the same name still raise.
- **Engine.** Keep every existing behavior and test.
  - `evaluate` also returns `predictions` and `targets`, lists of ints in loader order.
  - `run_epoch` and `fit` accept `scheduler=None` (stepped once per training batch) and `use_amp: bool = False` (autocast and gradient scaling on CUDA; a no-op on CPU).
  - New `macro_f1(confusion: list[list[int]]) -> float` and `mean_class_distance(confusion: list[list[int]]) -> float` (mean of |true index - predicted index| over all samples). A class with no predictions and no samples contributes F1 of 0.
  - Tests for each, with hand-computed values.
- **Makefile.** Replace the `colab-a3` target with a pattern target `colab-%` that depends on `notebooks` and runs `scripts/colab_run.py $*`. `make help` still lists it. `make -n colab-a3` and `make -n colab-a1` print the right command.
- **Activity 3 cells.** Renumber the files to a clean sequence with no letter suffix, keeping order. In the header, the time line says the measured time excludes machine start-up and can vary with download speed.
- Do not run Colab in this task. The delivered Activity 3 notebook becomes stale and `make check-notebooks` will name it; plan 5 reruns every notebook once.

Acceptance: `make test` green; `make notebooks` builds.

Commit messages: `Ignore comprehension variables when checking notebook names`, `Return predictions and support schedulers and mixed precision in the engine`, `Add generic Colab make target and renumber Activity 3 cells`.

### Task 2: Attention and Vision Transformer modules

**Files:** `src/dlv/attention.py`, `src/dlv/vit.py`, `tests/test_attention.py`, `tests/test_vit.py`.

**Produces:**

- `scaled_dot_product_attention(query, key, value, mask=None) -> tuple[Tensor, Tensor]`: returns output and attention weights. `mask` is boolean, true where attention is allowed. A row with nothing allowed yields zeros, not NaN.
- `MultiHeadAttention(embed_dim: int, num_heads: int, dropout: float = 0.0)`: each head owns its own query, key and value projections (separate `nn.Linear` layers per head, each mapping `embed_dim` to `embed_dim // num_heads`); head outputs are concatenated and passed through one output projection. `forward(x, return_weights=False)` returns the output, or `(output, weights)` with weights shaped `(batch, heads, tokens, tokens)`. Raises `ValueError` when `embed_dim` is not divisible by `num_heads`.
- `TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout=0.0)`: pre-norm. LayerNorm, multi-head attention, residual; LayerNorm, feedforward of two linear layers with GELU between them, residual. `forward(x, return_weights=False)`.
- `PatchEmbedding(image_size, patch_size, in_channels, embed_dim)`: image to sequence of patch tokens. Raises `ValueError` naming both sizes when the input size differs from `image_size`.
- `VisionTransformer(image_size=64, patch_size=8, num_classes=8, embed_dim=192, depth=6, num_heads=3, mlp_dim=768, dropout=0.1, use_positional_embedding=True)`: learnable class token, learnable positional embedding added to the token sequence, a stack of encoder blocks, final LayerNorm, linear head on the class token. `forward(images, return_attention=False)` returns logits, or `(logits, attention)` where attention is a list with one `(batch, heads, tokens, tokens)` tensor per block.
- `class_token_attention(attention: Tensor, grid_size: int) -> Tensor`: from one block's weights, the attention from the class token to every patch, shaped `(batch, heads, grid_size, grid_size)`.

**Tests, at least:**

- Attention weights are non-negative and each row sums to 1; output shape; a masked position gets weight 0; a fully masked row gives zeros with no NaN; with identical keys the weights are uniform.
- Multi-head: output shape; weights shape; the module holds `3 * num_heads` projection layers plus one output projection; changing the projections of one head changes that head's weights and leaves the other heads' weights unchanged; invalid `embed_dim` raises.
- Encoder block: shape preserved; with both sublayers' final projections zeroed the block is the identity; gradients reach every parameter.
- Patch embedding: `(2, 3, 64, 64)` with patch 8 gives `(2, 64, 192)`; wrong input size raises with both sizes in the message.
- ViT: logits `(batch, 8)`; class token and positional embedding are parameters that require gradients; `return_attention` gives `depth` tensors of shape `(batch, 3, 65, 65)`.
- Position: in eval mode with `use_positional_embedding=False`, shuffling the patches of the image (same permutation applied to patch positions) leaves the logits unchanged within 1e-5; with `use_positional_embedding=True` the logits change. This test backs the written explanation in the report.
- `class_token_attention` shape and that each head's map sums to less than or equal to 1.

Commit message: `Add attention, encoder block and Vision Transformer built from scratch`.

### Task 3: Paintings data module

**Files:** `src/dlv/paintings.py`, `tests/test_paintings.py`, `pyproject.toml` and `uv.lock` (add `pyarrow`).

**Produces:**

- `PERIODS`: eight `(label, first_year, last_year)` entries, labels in Portuguese: "até 1599", "1600 a 1799", "1800 a 1849", "1850 a 1874", "1875 a 1899", "1900 a 1924", "1925 a 1949", "1950 em diante".
- `parse_year(text) -> int | None`: first four-digit year between 1000 and 2029 found in the text; `None` otherwise.
- `period_of(year: int) -> int`: index into `PERIODS`.
- `download_wikiart(destination: Path) -> list[Path]`: lists the parquet files through the Hugging Face API and downloads each once (streamed, marker file as in `download_kaggle_dataset`). Returns the local paths sorted.
- `read_metadata(parquet_paths) -> dict[str, list]`: keys `year` (int or None), `artist`, `style`, `row` (global row number across files in path order). Reads only the `date`, `artist` and `style` columns.
- `select_balanced(labels: list[int | None], per_class: int, seed: int) -> list[int]`: indices of up to `per_class` rows per class, rows with `None` left out, deterministic, sorted.
- `split_by_group(groups: list[str], labels: list[int], fractions=(0.7, 0.15, 0.15), seed=42) -> tuple[list[int], list[int], list[int]]`: no group in more than one part; sizes close to the fractions; every class present in every part when the data allows; deterministic. An empty or unknown group value (`""`, `None`, any spelling of "unknown artist") makes that row its own group.
- `read_images(parquet_paths, rows: list[int]) -> list[bytes]`: encoded image bytes for the given global rows, in the order given, reading one row group at a time so memory stays near the size of the selection.
- `decode_images(images: list[bytes], size: int) -> Tensor`: uint8 tensor `(n, 3, size, size)`, RGB, resized with antialiasing.
- `EncodedImageDataset(images: list[bytes], labels: list[int], transform)`: decodes on access, for the 224x224 model.
- `accuracy_by_group(targets, predictions, groups, minimum=20) -> list[dict]`: one entry per group with at least `minimum` samples: `group`, `count`, `accuracy`, `mean_distance`; sorted by count, largest first.

**Tests, at least:** year parsing for `"1920"`, `"1997.0"`, `"c.1904"`, `"None"`, `""`, `"12th century"`, `"3000"`; period boundaries at 1599/1600 and 1949/1950; metadata, image reading and row numbering against small parquet files written in the test with two files and several row groups, image order preserved for unsorted `rows`; balanced selection counts, determinism and `None` handling; group split with one painter spanning several classes has no painter overlap, covers all indices once, is deterministic, and treats unknown painters as separate groups; download with a fake HTTP layer is skipped the second time and leaves no marker on failure; `decode_images` shape and dtype; `accuracy_by_group` against hand-computed values and the `minimum` filter.

Commit message: `Add paintings data module with painter-wise split`.

### Task 4: Figures for attention maps and group breakdown

**Files:** `src/dlv/figures.py`, `tests/test_figures.py`.

**Produces** (each saves a PNG and returns the `Figure`, labels in Portuguese):

- `plot_attention_maps(images, maps, titles, path)`: `images` is a uint8 tensor `(n, 3, h, w)`, `maps` a float tensor `(n, heads, g, g)`. One row per image: the image, then one panel per head with the map upsampled over the image. `titles` has one string per row.
- `plot_group_accuracy(entries, path, title)`: horizontal bars of accuracy per group from `accuracy_by_group`, with the count shown on each bar.
- `plot_histories(histories: dict[str, dict], path)`: loss and accuracy of several named runs on shared axes, validation curves only for accuracy and both curves for loss.
- `plot_class_counts(counts: dict[str, dict[str, int]], path)`: grouped bars of samples per class for each split.

Tests assert panel counts, labels and that files are written.

Commit message: `Add attention map, group accuracy and multi-run history plots`.

### Task 5: Activity 1 notebook and Colab run

**Files:** `notebooks_src/a1/*`, `scripts/build_notebooks.py` (`NOTEBOOKS` gains `"a1": "A1_vision_transformers"`), run outputs `notebooks/A1_vision_transformers.ipynb`, `results/a1.json`, `report/figures/a1_*.png`.

Follow the structure of `notebooks_src/a3/`: header with requirements, support modules, setup with `seed_everything` and `start_run("a1")`, sections with a short Markdown cell in Portuguese before each code cell, and a last cell that calls `save_run`.

**Data:** download; read metadata; label by `period_of(parse_year(date))`; `select_balanced` with 2,000 per class and seed 42; `split_by_group` on painter; read the selected images once; print counts per class and per split and the number of distinct painters per split, and confirm in a printed line that no painter is shared.

**Model A, from scratch:** `VisionTransformer` defaults (64x64, patch 8, embedding 192, 6 blocks, 3 heads). Images decoded once to 64x64 tensors held in memory. Training augmentation: random crop with padding 4 and horizontal flip. AdamW, learning rate 1e-3, weight decay 0.05, linear warm-up for the first 5 epochs then cosine decay, batch size 256, 60 epochs.

**Model A without position:** the same model and schedule with `use_positional_embedding=False`.

**Model B, pre-trained:** torchvision `vit_b_16` with `ViT_B_16_Weights.IMAGENET1K_V1`, head replaced by a linear layer with 8 outputs, all layers trained. Input: resize 256, centre crop 224, ImageNet mean and standard deviation (the preset of those weights). AdamW, learning rate 3e-5 for the backbone and 1e-3 for the new head, weight decay 0.05, batch size 32, mixed precision, 3 epochs.

**Evaluation on the test split, for each of the three models:** accuracy, macro F1, mean distance in periods, confusion matrix, parameters, training time in seconds. Accuracy within one period (|distance| <= 1).

**Attention:** for Model A, take six test paintings (three classified correctly and three incorrectly, chosen deterministically, covering different periods) and plot the class-token attention of every head of the last block. Also plot the mean over heads of the first block for the same paintings.

**Breakdown by art movement:** `accuracy_by_group` over test predictions of Model A and Model B with the `style` column, minimum 20 samples.

**Metrics file keys:** `periods`, `class_counts`, `split_sizes`, `painters_per_split`, `rows_total`, `rows_with_year`, `hyperparameters`, `models` (per model: `parameters`, `train_seconds`, `history`, `accuracy`, `within_one_period`, `macro_f1`, `mean_distance`, `per_class_accuracy`, `confusion`), `by_style` (per model), `attention_examples` (title, painter, year, true period, predicted period for each plotted painting).

**Figures:** class counts, histories of the three runs, confusion matrix of Model A and of Model B, attention maps (last block per head; first block mean), accuracy by movement for Model A and Model B.

**Process:**

1. `make notebooks`, then `uv run pytest tests/test_notebooks.py -q`.
2. `make colab-a1`. Budget: 5 Colab runs. Read the result. A run that fails or exceeds 30 minutes is a defect to fix in `src/dlv` or the cells, with a test when the fix is in `src/dlv`.
3. If the notebook takes more than 25 minutes, reduce cost in this order and record what you did: Model B epochs 3 to 2; scratch epochs 60 to 40; per-class sample 2,000 to 1,500. Do not change anything to raise a score.
4. Set the header's memory and time lines from the measured values of the final run, rebuild, and confirm with one more run that the delivered notebook equals a fresh build (`make check-notebooks` must not name `A1_vision_transformers`).
5. Look at the attention figures with the Read tool and note, in the task report, what each head appears to attend to. These notes feed Task 6.

Acceptance: executed notebook with no error outputs; all figures present; metrics file has every key; printed line confirms zero shared painters; total time under 30 minutes.

Commit messages: one per fix in `src/dlv` if any, then `Add Activity 1 notebook with Colab T4 run`.

### Task 6: Activity 1 report section

**Files:** `report/secoes/10_atividade1.md`, `notebooks_src/a1/<analysis cell>.md` (short reading of results placed before the last cell), then rerun and rebuild.

Read `results/a1.json`, every `report/figures/a1_*.png`, the notes from Task 5, and `report/secoes/30_atividade3.md` as the model for tone and structure.

Headings, in this order, and what each must hold:

- `# Atividade 1: Datação de pinturas com Vision Transformers`
- `## Problema`: the task, the dataset and its source, rows with usable year, the eight periods with counts, why the split is by painter, split sizes and painters per split. State that style is not date: a painter can work in an older style.
- `## Decisões técnicas`
  - `### Attention e encoder construídos do zero` (rubric: scaled dot-product and multi-head attention as testable modules, separate projections per head, concatenation; encoder block with two-layer feedforward, LayerNorm and residual connections): what was built, where, how it is tested.
  - `### Vision Transformer` (rubric: patch embedding, learnable class token, positional encoding, logits): the pipeline from image to logits, with the numbers of this model (64 patches, 65 tokens, parameters).
  - `### Por que attention precisa de positional encoding` (rubric): the argument that attention treats its input as a set, the test that shows it, and the measured result of the model without position.
  - `### Hiperparâmetros`: each choice with its reason, for both models.
- `## Resultados`: comparison table of the three models (accuracy, within one period, macro F1, mean distance, parameters, training time); curves; confusion matrices; reading of where errors fall (neighbouring periods or far ones).
- `## O que o modelo observa` (rubric: attention heatmap for at least one head, written interpretation, emerging regions): the attention figures and what each head attends to, painting by painting. Report only what the figures show. If a head shows no clear pattern, say so.
- `## Erros por movimento artístico`: the breakdown, which movements are dated worst and a plausible reason tied to how long each movement lasted.
- `## ViT do zero ou ViT pré-treinado` (rubric: quantitative table, choice justified by the data): which one to use for this domain and why, from the numbers.
- `## Pré-treinamento: BERT e ViT` (rubric: differences, what each strategy maximizes): masked language modelling and next sentence prediction against supervised classification on large labelled image sets; mention masked image modelling (MAE, BEiT) as the closest image counterpart of BERT.
- `## DeiT e Swin Transformer` (rubric: what each solves that the original ViT does not): data efficiency through distillation and strong augmentation; hierarchy, local windows and linear cost in image size.
- `## Quando usar ViT e quando usar CNN` (rubric, tied to this domain): inductive bias, data volume, global composition against local texture, with the Activity 3 result as the CNN reference point where it helps.
- `## Análise crítica e próximos passos`: limits (single seed, 64x64 input for the scratch model, label noise in dates, sample of 16,000 from 80,000) and what would be changed.

Then: rerun `make colab-a1` only if a cell file changed in a way that alters the built notebook (the analysis cell does), `make report`, read the PDF pages, `make zip`.

Budget: 2 Colab runs.

Commit message: `Add Activity 1 report section`.
