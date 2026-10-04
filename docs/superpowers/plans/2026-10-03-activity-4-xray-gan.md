# Activity 4: Chest X-ray Case Study with a Conditional GAN - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Deliver the Activity 4 notebook and report sections: rebuild the failed X-ray project, train a conditional GAN to generate COVID-19 images, measure the effect of synthetic images on COVID-19 recall, and write the two case studies.

**Architecture:** New modules `xray.py` and `gan.py` in `src/dlv`, tested on CPU. Notebook built from `notebooks_src/a4/`, run on a Colab T4 by the existing runner.

**Tech Stack:** Python 3.13, PyTorch 2.11, torchvision 0.26, scikit-learn, matplotlib, pytest, Colab CLI.

**Spec:** `docs/superpowers/specs/2026-10-03-computer-vision-project-design.md`

This is plan 3 of 5. It states interfaces, tests and acceptance checks; the implementer writes the code. Global constraints are the ones in `2026-10-03-activity-1-vision-transformers.md`, section "Global Constraints", and bind this plan too.

## Facts already verified

- Dataset: Kaggle `tawsifurrahman/covid19-radiography-database`, 0.8 GB, no login, through `download_kaggle_dataset` in `src/dlv/data.py`.
- Layout: `COVID-19_Radiography_Dataset/<class>/images/*.png` and `<class>/masks/*.png`. Classes: `COVID` (3,616 images), `Lung_Opacity` (6,012), `Normal` (10,192), `Viral Pneumonia`. Masks are lung segmentation masks, not X-rays, and must never be loaded as images.
- The assignment's failed project: 1,200 images, Normal 840, Pneumonia 240, COVID-19 120; 80/20 split with no stratification; ResNet-18 with no pre-training; SGD, fixed learning rate 0.01, 15 epochs, no augmentation; global accuracy only; 93% training accuracy, 61% validation accuracy.
- `list_image_folder` in `data.py` labels by the immediate parent folder, which here would be `images` or `masks`. This plan uses its own listing.

## Review Focus

1. A mask file is loaded as an X-ray: listing must take files only from `<class>/images/`.
2. A generated image is used for evaluation: validation and test sets hold real images only, and a test proves the mixing function never touches them.
3. The GAN sees validation or test images: it is trained on the training split only.
4. The same file content appears in training and in test: exact copies are removed before sampling, as Activity 3 taught.
5. The generator is asked for a class index outside the known range: a clear error.

---

### Task 1: X-ray data module

**Files:** `src/dlv/xray.py`, `tests/test_xray.py`.

**Produces:**

- `XRAY_CLASSES`: tuple of `(folder, label)` pairs in fixed order: `("Normal", "Normal")`, `("Viral Pneumonia", "Pneumonia")`, `("COVID", "COVID-19")`.
- `list_xray_images(root: Path) -> tuple[list[Path], list[int]]`: readable images under `<folder>/images/` for the three classes, sorted; label is the index into `XRAY_CLASSES`; other folders and all masks ignored; exact copies by file content removed (keep the first path; a content found under two classes is left out entirely). Reuse helpers from `dlv.data` through a single-statement import; do not redefine their names.
- `sample_per_class(labels: list[int], counts: dict[int, int], seed: int, exclude: list[int] = ()) -> list[int]`: sorted indices with exactly `counts[c]` items of each class, never choosing an index in `exclude`; raises `ValueError` naming the class when not enough items exist.
- `load_grayscale(paths: list[Path], size: int) -> Tensor`: uint8 `(n, 1, size, size)`, resized with antialiasing.
- `class_report(confusion: list[list[int]]) -> list[dict]`: per class `recall`, `precision`, `f1`, `support`, in class order; zero denominators give 0.0.
- `mix_synthetic(real_images: Tensor, real_labels: list[int], synthetic_images: Tensor, synthetic_label: int) -> tuple[Tensor, list[int]]`: training set with the synthetic images appended; inputs unchanged.
- `summarise_runs(reports: list[list[dict]], metric: str) -> list[dict]`: per class mean and standard deviation of a metric across seeds.

**Tests, at least:** listing ignores masks and other class folders, labels by class folder, drops exact copies and cross-class copies; sampling counts, determinism, exclusion and the error message; `load_grayscale` shape and dtype for RGB, RGBA and grayscale inputs; `class_report` against hand-computed values including a class never predicted; `mix_synthetic` shapes, labels and untouched inputs; `summarise_runs` against hand-computed mean and standard deviation.

Commit message: `Add X-ray data module`.

### Task 2: Conditional GAN module

**Files:** `src/dlv/gan.py`, `tests/test_gan.py`.

**Produces:**

- `ConditionalGenerator(latent_dim: int = 100, num_classes: int = 3, base_channels: int = 64)`: `forward(noise, labels)` returns `(n, 1, 64, 64)` in the range -1 to 1. Conditioning through a label embedding combined with the noise. Raises `ValueError` for a label outside the range.
- `ConditionalDiscriminator(num_classes: int = 3, base_channels: int = 64, spectral_norm: bool = False)`: `forward(images, labels)` returns one logit per image. With `spectral_norm=True` every convolution and linear layer is wrapped in spectral normalisation and no batch normalisation is used.
- `discriminator_loss(real_logits, fake_logits, kind: str) -> Tensor` and `generator_loss(fake_logits, kind: str) -> Tensor` for `kind` in `"bce"` and `"hinge"`; `ValueError` for anything else.
- `pairwise_diversity(images: Tensor) -> float`: mean absolute difference between all pairs of images in a batch scaled to 0 to 1. Zero for identical images. The collapse indicator.
- `nearest_real_distance(synthetic: Tensor, real: Tensor) -> Tensor`: for each synthetic image, the mean absolute difference to its closest real image. The memorisation indicator.
- `train_gan(generator, discriminator, images: Tensor, labels: Tensor, epochs: int, batch_size: int, device, loss: str = "bce", generator_lr: float = 2e-4, discriminator_lr: float = 2e-4, betas=(0.5, 0.999), seed: int = 42, snapshot_epochs=(), snapshot_noise=None, snapshot_labels=None) -> dict`: adversarial training with alternating updates, one discriminator step and one generator step per batch. `images` are uint8 `(n, 1, 64, 64)`; scaling to -1..1 happens inside. Returns a history with per-epoch `discriminator_loss`, `generator_loss`, `real_score` (mean sigmoid of real logits), `fake_score`, `diversity` (per class, from fixed noise), and `snapshots` (epoch to uint8 image grid tensor for the snapshot noise and labels).
- `generate(generator, labels: Tensor, device, seed: int) -> Tensor`: uint8 `(n, 1, 64, 64)`.

**Tests, at least:** generator output shape and range; changing only the label changes the output; out-of-range label raises; discriminator output shape; spectral variant has spectral normalisation on its weight layers and no batch norm; loss values against hand-computed numbers for both kinds; unknown kind raises; diversity is zero for identical images and positive otherwise; nearest-real distance is zero when the synthetic image is a real one; one epoch of `train_gan` on 16 random images changes parameters of both networks, records every history key, and is reproducible for the same seed; snapshots have the requested epochs; `generate` shape, dtype and determinism.

Commit message: `Add conditional GAN with standard and hinge losses`.

### Task 3: Activity 4 notebook and Colab run

**Files:** `notebooks_src/a4/*`, `scripts/build_notebooks.py` (`NOTEBOOKS` gains `"a4": "A4_estudo_caso_raio_x"`), `src/dlv/figures.py` and its test (new plots), run outputs `notebooks/A4_estudo_caso_raio_x.ipynb`, `results/a4.json`, `report/figures/a4_*.png`.

Follow the structure of `notebooks_src/a1/` and `a3/`.

**Data:** download; `list_xray_images`; print files found, copies removed and images per class. Draw the project set with `sample_per_class` (840, 240, 120; seed 42). Draw a separate real test set of 300 images per class from the remaining images (`exclude` the project set). State in the notebook why: the project set leaves only a handful of COVID-19 images for testing, and the public database has thousands more real ones, so recall can be measured on 300 per class instead of about 20. Images loaded at 64x64 grayscale.

**Part A, the failed project reproduced:** 80/20 split of the project set with no stratification (plain shuffle, seed 42); ResNet-18 with random weights; SGD, learning rate 0.01, no momentum, 15 epochs, batch size 32, no augmentation. Report what the original team reported (training and validation accuracy) and what they did not: recall per class on their validation split and on the real test set, and the number of COVID-19 images that landed in their validation split.

**Part B, protocol for the rest of the notebook:** stratified 80/20 split of the project set into training and validation; the real test set stays apart.

**Part C, the GAN:** trained on the training split only, all three classes, 64x64.
- Version 1: `ConditionalDiscriminator(spectral_norm=False)`, loss `"bce"`, both learning rates 2e-4.
- Version 2: `spectral_norm=True`, loss `"hinge"`, discriminator learning rate 4e-4, generator learning rate 1e-4.
- Same epochs, batch size, seed and snapshot noise for both. Start with 300 epochs and batch size 64.
- Diagnose each version from its curves, its snapshot grids and its diversity per class: say whether there is mode collapse, divergence or neither, with the evidence.
- If version 2 does not improve on version 1 by the diversity score and by eye, try in this order and report each attempt honestly: WGAN-GP (add it to `gan.py` with tests); differentiable augmentation of real and fake images before the discriminator (random translation and brightness, added to `gan.py` with tests). If nothing improves, say so; do not present a failure as a success.
- Memorisation check for the chosen version: `nearest_real_distance` of generated COVID-19 images against training COVID-19 images, compared with the same distance between real training and real validation COVID-19 images.

**Part D, the recall experiment:** classifier is torchvision `resnet18` with ImageNet weights, all layers trained, input 64x64 grayscale repeated to three channels and resized to 128, ImageNet normalisation, Adam 1e-4, 10 epochs, batch size 32, light augmentation (horizontal flip off, since left and right matter in a chest X-ray; small random translation and rotation up to 7 degrees) identical in every arm. Three seeds per arm.
- Arm 1: real training images only.
- Arm 2: real plus synthetic COVID-19 images from the chosen GAN version, enough to match the Normal count.
- Arm 3: real only, with class-weighted loss. The simple alternative that needs no GAN.
- Evaluate each run on the real test set: recall, precision and F1 per class, and the confusion matrix. Report mean and standard deviation over seeds.

**Metrics file keys:** `files_found`, `duplicates_removed`, `images_per_class`, `project_counts`, `test_counts`, `original_project` (reported accuracies, recalls per class on its own validation and on the real test, COVID-19 images in its validation split), `gan` (per version: hyperparameters, history without snapshots, final diversity per class, diagnosis flags you computed), `memorisation`, `arms` (per arm: per-seed reports and the summary), `hyperparameters`.

**Figures:** class counts; original project curves; GAN loss and score curves for both versions; snapshot grids over epochs for both versions; diversity per class over epochs for both versions; real against synthetic COVID-19 samples; recall per class per arm with error bars; confusion matrix of one representative run per arm.

**Process:** as in Activity 1 Task 5: check locally what can be checked on tiny made-up data; run with `make colab-a4` in the background with output redirected under `build/`; budget 5 Colab runs; under 30 minutes; header values from the final run; `make check-notebooks` must not name this notebook. Look at every figure with the Read tool and record what the snapshot grids show.

Commit messages: one per addition to `src/dlv`, then `Add Activity 4 notebook with Colab T4 run`.

### Task 4: Report sections for the X-ray and traffic case studies

**Files:** `report/secoes/40_atividade4_raio_x.md`, `report/secoes/45_atividade4_trafego.md`, an analysis cell in `notebooks_src/a4/` before the last cell.

Read `results/a4.json`, every `report/figures/a4_*.png`, the Task 3 notes, the two references of the assignment (the Scientific Reports article at https://www.nature.com/articles/s41598-023-37743-4 and the page at https://www.meegle.com/en_us/topics/transfer-learning/transfer-learning-for-traffic-analysis), and `report/secoes/30_atividade3.md` for tone. Cite only what you actually read; if a page cannot be fetched, say so in the task report and do not quote it.

`40_atividade4_raio_x.md`, headings in this order:

- `# Atividade 4.1: Triagem de COVID-19 em radiografias de tórax`
- `## O projeto analisado`: what the earlier team did and reported.
- `## Diagnóstico: problemas técnicos e impacto clínico` (rubric: at least five problems, each with its clinical impact): one subsection per problem. Use the reproduction in Part A as evidence where it applies. Cover at least: class imbalance left untreated; split with no stratification; global accuracy as the only metric; no pre-training on a small set; fixed learning rate and optimiser choice; no augmentation; overfitting visible in the 93% against 61% gap; no independent test set; possible duplicated images and patient overlap.
- `## A base de dados`: counts, copies removed, how the project set and the real test set were drawn.
- `## GAN condicional para a classe COVID-19` (rubric: conditional GAN with a correct adversarial training loop): architecture, conditioning, the loop.
- `## Instabilidade do treinamento e mitigação` (rubric: diagnosis of mode collapse or divergence, at least one mitigation, evidence of improvement): what version 1 showed, what was changed, what version 2 showed, with the figures and the diversity numbers. If the mitigation did not help, say so and show it.
- `## As imagens geradas são cópias?`: the memorisation check.
- `## Efeito das imagens sintéticas na recall` (rubric: with and without synthetic images): the three arms, mean and spread, and a plain statement of whether the difference is larger than the spread between seeds.
- `## Plano de melhoria integrado` (rubric: model, metric, synthetic augmentation, criterion for clinical adoption): one plan that addresses every diagnosed problem, with a concrete adoption criterion (for example a minimum COVID-19 recall at a fixed specificity on an external set, prospective validation, a radiologist in the loop).
- `## Limites deste estudo`.

`45_atividade4_trafego.md`:

- `# Atividade 4.2: Transfer learning para análise de tráfego urbano`
- `## O projeto analisado`
- `## Problemas técnicos e metodológicos` (rubric: at least four, each with operational impact): one subsection per problem, each ending with how to address it. Cover at least: validation frames from the same cameras as training, so 78% says nothing about new cameras; domain shift for rain and night; 800 frames across 12 cameras is small and correlated in time; ImageNet features recognise objects, while flow is a property of the whole scene and of density; class boundaries between "moderado" and "congestionado" are subjective; no monitoring after deployment.
- `## Riscos de partir da ImageNet para classificar fluxo` (rubric).
- `## Como eu conduziria o projeto`.

Then rerun the notebook if a cell changed, `make report`, read the PDF pages, `make zip`. Budget: 2 Colab runs.

Commit message: `Add Activity 4 report sections`.
