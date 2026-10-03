# Computer Vision course project (INFNET 26E3_3) - design

Date: 2026-10-03. Deadline: 2026-10-12 23:59. Source of truth for requirements: `req/requirements.html`.

## Goal

Deliver four notebooks and one PDF report that earn all 26 pass/fail rubric items.

- Each notebook runs start to finish on Colab with a T4 GPU, with no edits.
- Each notebook states memory needs and estimated run time at the top.
- The report is one PDF in Portuguese covering the four activities.
- AI use is cited in the report.

## Decisions already taken

- Activity 1 domain: dating paintings by time period.
- Activity 2 model: CLIP used as is, no training.
- Activity 3 model: ResNet-50, frozen, new last layer.
- Activity 4.1 approach: class-conditional GAN, a basic version and a stabilized version.
- Code in English. Report and notebook prose in Portuguese.
- Tooling: `uv`, `make`, Colab CLI (`colab`, version 0.7.4, verified on a free account with a T4).

## Repository layout

```
src/dlv/              shared package (English)
  attention.py        scaled dot-product attention, multi-head attention
  vit.py              patch embedding, encoder block, ViT
  data.py             dataset download, cleaning, splits
  engine.py           training loop, evaluation, metrics
  clip_search.py      CLIP embeddings, ranking, text search
  gan.py              conditional generator and discriminator, GAN training loop
  figures.py          every plot used by notebooks and report
tests/                CPU-only tests with tiny random inputs
scripts/
  build_notebooks.py  writes the four notebooks, copying src/dlv code into cells
  build_report.py     joins report/secoes/*.md and renders the PDF
  colab_run.sh        runs one notebook on a Colab T4 and downloads results
notebooks/            generated, never edited by hand
  A1_vision_transformers.ipynb
  A2_clip_ads16.ipynb
  A3_cnn_kaggle.ipynb
  A4_estudo_caso_raio_x.ipynb
results/              metrics as JSON, one file per activity
report/
  secoes/*.md         report sources in Portuguese
  figures/            PNG figures produced by the runs
  diagrams/           Excalidraw sources and exported PNGs
  anderson_correa_deep-learning-and-vision_computer-vision.pdf
Makefile
pyproject.toml, uv.lock
```

Rules for this layout:

- `src/dlv` is the only place where logic lives. Notebooks are build output.
- Each notebook is self-contained. It installs its packages, downloads its data and defines its code in cells. It imports nothing from this repository.
- Every run writes metrics to `results/` and figures to `report/figures/`. The report reads only those files.
- Local Python and torch versions match Colab (Python 3.13, torch 2.11).

## Make targets

- `make setup`: `uv sync`
- `make test`: run `tests/` on CPU
- `make notebooks`: regenerate the four notebooks from `src/dlv`
- `make colab-a1`, `colab-a2`, `colab-a3`, `colab-a4`: run one notebook on a Colab T4, download `results/` and figures, stop the machine
- `make report`: build the PDF
- `make zip`: build `anderson_correa_computer-vision_pd.zip` with notebooks and PDF
- `make help`

## Data sources

All four download with no login.

| Activity | Dataset | Source | Size |
| --- | --- | --- | --- |
| 1 | WikiArt, 256x256, with `date` and `style` | Hugging Face `Artificio/WikiArt` | 1.7 GB |
| 2 | ADS-16 | Kaggle `groffo/ads16-dataset` | 0.8 GB |
| 3 | Images Dataset | Kaggle `pavansanagapati/images-dataset` | 0.75 GB |
| 4.1 | COVID-19 Radiography Database | Kaggle `tawsifurrahman/covid19-radiography-database` | 0.8 GB |

Kaggle files come from `https://www.kaggle.com/api/v1/datasets/download/<owner>/<name>`, which redirects to a public archive.

## Activity 1: dating paintings with Vision Transformers

Problem: given a painting, predict the period when it was painted.

Data preparation:

- Parse the year from `date` with one regular expression. Drop rows with no year (about 22%).
- Eight classes. Default boundaries, to be confirmed by Anderson during implementation:
  before 1600, 1600 to 1799, 1800 to 1849, 1850 to 1874, 1875 to 1899, 1900 to 1924, 1925 to 1949, 1950 onward.
- Sample 2,000 paintings per class, 16,000 in total, with a fixed seed.
- Split by painter, 70/15/15, so no painter appears in two splits.

Models:

- Model A, built by hand from `attention.py` and `vit.py`, trained from random weights.
  - Input 64x64, patch size 8, 64 patches plus one class token.
  - Small configuration: embedding 192, 6 blocks, 3 heads.
  - Learnable position embedding.
- Model B, torchvision `vit_b_16` with ImageNet weights.
  - Input 224x224, classification head replaced by 8 outputs, fine-tuned for a few epochs with a low learning rate.

Outputs:

- Table comparing A and B: accuracy, macro F1, mean distance in periods, parameters, training time.
- Confusion matrix for each model.
- Attention heat maps from Model A for at least one head, on several paintings.
- Error breakdown by art movement, using the `style` column the models never saw.
- Short ablation: Model A with position embedding removed, to support the written explanation.

Written sections in the report:

- Why attention alone ignores position, and what the position embedding adds.
- BERT pre-training versus ViT pre-training, and what each one optimizes.
- DeiT and Swin: what each solves that the original ViT does not.
- When transformers beat CNNs and when CNNs are the better choice, for this domain.
- Justification of every hyperparameter, and what would be changed next.

Tests: attention output shapes, attention rows sum to one, heads use separate projections, encoder block keeps shape, ViT returns `(batch, 8)` logits, painter split has no overlap.

## Activity 2: semantic search in ads with CLIP

Model: `openai/clip-vit-base-patch32` through `transformers`. No training.

Pipeline:

- Use every ADS-16 image if the count is at least 500. Otherwise the plan stops and the gap is reported.
- Compute image embeddings once and cache them.

2.1 Ranking:

- At least 20 text prompts for objects and concepts.
- Cosine similarity between every image and every prompt.
- A concept counts as present in an image when its score passes a threshold. The threshold rule is chosen during implementation and justified with the score distribution.
- Report mean score and frequency for each concept.
- Show the five most frequent concepts with example images.

2.2 Text search:

- Function that returns the five closest images for a text query.
- At least eight queries, from generic to specific and from concrete to abstract.
- For each query, a short written analysis of what came back.

Written sections: how contrastive pre-training aligns text and images; CLIP text input compared with BERT tokenization, including padding and attention mask (shown with the real tokenizer output).

Tests: embeddings are unit length, search returns five distinct indices, ranking is sorted.

## Activity 3: object classifier with a pre-trained CNN

- Classes are read from the folder names of the dataset.
- Stratified split 70/15/15.
- ResNet-50 with ImageNet weights, every backbone weight frozen, `fc` replaced by a new layer with one output per class.
- One training run of the new layer only. Inputs normalized with ImageNet mean and standard deviation.
- Outputs: loss and accuracy curves per epoch, accuracy per class, overall accuracy, confusion matrix.

Written sections: at least three augmentation strategies judged class by class (where each helps, where each distorts); feature extraction versus fine-tuning given dataset size and domain; why ResNet-50 fits the T4 and the number of classes.

Tests: only `fc` weights require gradients, head output size equals the class count.

## Activity 4.1: chest X-ray case study

Rebuild the failed project:

- Sample Normal 840, Pneumonia 240, COVID-19 120 from the public database.
- Stratified split, with a test set of real images only.

GAN:

- Conditional GAN on 64x64 grayscale images, trained only on the training split.
- Version 1: conditional DCGAN with the standard loss. Expected to show mode collapse or divergence.
- Version 2: same network with spectral normalization in the discriminator, hinge loss and different learning rates for the two networks.
- Evidence of improvement: loss curves, sample grids over epochs, and a diversity score for generated COVID-19 images.

Recall experiment:

- Same classifier in both arms: ResNet-18 with ImageNet weights.
- Arm A: real training images only.
- Arm B: real training images plus generated COVID-19 images.
- Three seeds per arm. Report recall, precision and F1 per class, with mean and spread.

Written sections: at least five technical problems of the original project with the clinical impact of each; integrated improvement plan covering model, metric, synthetic augmentation and criterion for clinical adoption.

Tests: generator output shape, discriminator output shape, labels condition both networks, one training step changes both networks.

## Activity 4.2: traffic case study

Report only. At least four technical and methodological problems, the operational impact of each, how to address each, and the risks of moving from ImageNet to traffic flow classification.

## Report

- Sources in `report/secoes/`, one Markdown file per activity, in Portuguese, simple language, bullet points.
- Each activity section has: problem definition, technical decisions with justification, results with metrics and figures, critical analysis.
- One Excalidraw diagram per activity, exported to PNG.
- A final section listing sources and AI tools used.
- Output file: `report/anderson_correa_deep-learning-and-vision_computer-vision.pdf`.

## Rubric coverage

| Items | Where |
| --- | --- |
| 1, 2 | A3 notebook |
| 3, 4, 5 | A3 report section |
| 6, 8, 11 | `attention.py`, `vit.py`, tests, A1 notebook |
| 7, 12 | A1 heat maps and written reading |
| 9 | A1 ablation and report |
| 10, 14, 15 | A1 report section |
| 13, 16 | A1 comparison table |
| 17, 20 | A2 report section |
| 18, 19 | A2 notebook |
| 21, 26 | A4.1 report section |
| 22, 23, 24 | A4 notebook |
| 25 | A4.2 report section |

## Order of work

1. Scaffold and Activity 3, to prove the whole chain: download, train on T4, download results, build notebook, build report.
2. Activity 1.
3. Activity 4.1.
4. Activity 2.
5. Report text, diagrams, one browser run of each notebook on Colab, zip.

Activities 2, 3 and 4 touch separate modules, so they can be built in parallel once the scaffold exists.

## Code written by Anderson

Three small pieces are left for Anderson, with the surrounding file prepared:

- The scaled dot-product attention function in `attention.py`.
- The period boundaries in `data.py`.
- The threshold rule in `clip_search.py`.

## Risks

- ADS-16 image count is not confirmed. Checked first when Activity 2 starts.
- Free Colab quota for long runs is unknown. Each notebook targets under 30 minutes.
- `Artificio/WikiArt` is a third-party upload with no license stated. Use is non-commercial study, cited in the report.
- The stabilized GAN must improve visibly on the basic one. If hinge loss and spectral normalization are not enough, WGAN-GP is the fallback.
- The Colab CLI is new. If it fails, notebooks are run by hand in the browser and nothing else changes.

## Out of scope

- Training a second Activity 1 classifier on art movements.
- CycleGAN.
- Any implementation for Activity 4.2.
- Experiment tracking tools and deployment.
