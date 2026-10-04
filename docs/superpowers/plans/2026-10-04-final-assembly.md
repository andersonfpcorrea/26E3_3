# Final Assembly - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Turn four finished notebooks and five report sections into the deliverable: every notebook rerun once on a Colab T4 and matching a fresh build, one polished PDF report with diagrams and an AI-use section, and the delivery archive.

**Architecture:** No new modules. Small fixes to plots, the report builder and notebook text; one polish pass over the report; diagrams; one sequential rerun of all notebooks; a number check of every section against the final results files.

**Tech Stack:** as in the earlier plans.

**Spec:** `docs/superpowers/specs/2026-10-03-computer-vision-project-design.md`

This is plan 5 of 5. Global constraints are the ones in `2026-10-03-activity-1-vision-transformers.md`, section "Global Constraints". Two changes ruled during execution bind this plan: report tasks do not rerun notebooks except Task 5; cells may hold short orchestration functions (the spec rule that all logic lives in `src/dlv` is relaxed to models, data handling, metrics and plots).

Input for Tasks 1 and 2: `.superpowers/sdd/report-polish-list.md`, the list of non-blocking findings collected from every task review.

---

### Task 1: Plot, builder and notebook-text fixes

**Files:** `src/dlv/figures.py`, `src/dlv/clip_search.py`, `scripts/build_report.py`, their tests, notebook cells under `notebooks_src/` (headers and wording only), `Makefile`.

- **Plots**, each with a test: the legend of `plot_class_counts` does not cover bars; the first-block attention figure's title does not collide with panel headings; the GAN diversity figure's legend does not cover the real-diversity line; the recall figure draws the per-seed points over the bars and its axis says "Recall"; accuracy axes in `plot_group_accuracy` have ticks from 0 to 1.
- **CLIP listing:** the Activity 2 data cell gets the entries and the counts from one scan of the files.
- **Report builder:** images keep their aspect ratio and take a width chosen per figure: full text width for grids, snapshot sheets and multi-panel curves, about two thirds for single charts; the rule is by image aspect ratio or by an explicit width given in the Markdown image line, documented in the module docstring. A heading is never left alone at the foot of a page (keep headings with the content that follows). `resolve_link` resolves against the sections folder's parent that is passed in, not a global. The CSS warning about `border-collapse` no longer prints. The PDF test asserts that text with accents is present in the output, not only the file header.
- **Makefile:** `make report` works from a clean checkout; `make zip` puts the four notebooks and the PDF in `entrega/anderson_correa_computer-vision_pd.zip`; a new target `colab-all` runs `a3`, `a2`, `a1`, `a4` in sequence and stops at the first failure.
- **Notebook text** (Markdown cells only, no logic): each header states that the measured time excludes machine start-up and may vary with download speed, as a range where the download is large (Activity 1: "de 15 a 19 minutos"); "revocação" becomes "recall" wherever it appears in cells and figure labels; the Activity 2 data cell label "arquivos ilegíveis" becomes "arquivos de imagem que não abriram ou não são imagem".
- Do not move logic between cells and modules. Do not change any model, split, threshold, seed or hyperparameter.

Acceptance: `make test` green; `make notebooks` builds all four; `make report` builds with no warning.

### Task 2: Report polish

**Files:** `report/secoes/10_atividade1.md`, `20_atividade2.md`, `30_atividade3.md`, `40_atividade4_raio_x.md`, `45_atividade4_trafego.md`, `00_identificacao.md`, and the analysis cells `notebooks_src/a1/20_analysis.md`, `notebooks_src/a2/16_analysis.md`, `notebooks_src/a4/16_analysis.md`.

Apply every item of the polish list that concerns report text and analysis cells: the wording corrections with the replacement text given there, the removal of unsourced statements, the conditional wording for guesses, the terminology rule, and the cuts to repetition. Keep every heading, every rubric item and every finding. Add no claim and no number. Activity 3's section is checked against the same terminology rule and for the image-resize value (232).

Length targets after the cuts: Activity 1 at or under 6,000 words; Activity 4.1 at or under 4,300; the others unchanged or shorter.

Acceptance: mechanical checks pass (no en or em dashes, none of the banned words, headings unchanged); a script confirms no number was introduced that was not already in the section or in the polish list.

### Task 3: Diagrams, AI-use section and README

**Files:** `report/diagrams/*.excalidraw` and exported `*.png`, `report/secoes/*.md` (one image line each), `report/secoes/90_uso_de_ia_e_referencias.md`, `README.md`.

- One diagram per activity (1, 2, 3, 4.1) showing the pipeline: data, what is removed or split, models, what is measured. Made with the Excalidraw diagram skill, exported to PNG, placed after the "Problema" heading of each section. Labels in Portuguese, few words, readable at page width.
- `90_uso_de_ia_e_referencias.md`: the assignment requires every source and every AI tool to be cited. State which AI tool was used (Claude Code, by Anthropic), for what (planning, writing and reviewing code, running experiments, drafting and reviewing the report text), how the output was checked (automated tests, independent reviews of each task, every number checked against the results files, figures inspected), and that the student reviewed the work and is responsible for it. Then the references: datasets with their sources and licences as stated on their pages, pre-trained models and weights, libraries, the papers behind the methods named in the report (Transformer, ViT, DeiT, Swin, BERT, CLIP, GAN, conditional GAN, DCGAN, spectral normalisation, WGAN-GP, differentiable augmentation, ResNet), and the two references given in the assignment as they were found. Cite only what is real and what the report uses; give authors, title, venue and year; no invented page numbers or DOIs.
- `README.md` in Portuguese: what the project is, how to run (`make setup`, `make test`, `make notebooks`, `make colab-<atividade>`, `make report`, `make zip`), where each deliverable is, the structure of the repository.

### Task 4: Final reruns and number check

One agent owns Colab for this task.

1. `make test`, `make notebooks`.
2. Run the four notebooks in sequence on the T4 (`a3`, `a2`, `a1`, `a4`), one at a time, each in the background with output redirected, confirming no session is left after each.
3. For each notebook: no error outputs; `results/<activity>.json` has its keys; figures regenerated; header memory and time lines agree with the measured values (correct the header and rerun that notebook only if a line is off by more than the stated tolerance: time outside the stated range or more than 20% off a single value; memory more than 0.5 GB off).
4. `make check-notebooks` exits 0.
5. Number check: for each report section and each analysis cell, a script extracts every number and compares it with the final results file of that activity; every mismatch is corrected in the text to the value in the file, and every qualitative statement that depends on a number that changed is reread against the figures and results and corrected. Statements that no longer hold are rewritten to what the final run shows, never the other way round.
6. `make report`; read every page of the PDF; `make zip`; list the archive.

### Task 5: Whole-branch review, fixes, merge

Whole-branch review of everything since the last merge into `master`, one fix wave, one scoped re-review, then the owner decides how to integrate.
