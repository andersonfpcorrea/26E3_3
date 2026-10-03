# Foundation and Activity 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the project scaffold and deliver Activity 3 end to end: a self-contained notebook that trains a frozen ResNet-50 on a Colab T4, with results pulled back into the repository and shown in a PDF report.

**Architecture:** Logic lives in the `src/dlv` package and is tested on CPU. A build script turns a folder of cell files plus package modules into one self-contained notebook. A runner script executes that notebook on a Colab T4 through the `colab` CLI, checks for failed cells, and downloads metrics and figures. The report is built from Markdown sections that read those files.

**Tech Stack:** Python 3.13, uv, make, PyTorch 2.11, torchvision 0.26, scikit-learn, matplotlib, nbformat, pytest, markdown + xhtml2pdf, Google Colab CLI 0.7.4.

**Spec:** `docs/superpowers/specs/2026-10-03-computer-vision-project-design.md`

This is plan 1 of 5. Later plans (Activity 1, Activity 4.1, Activity 2, final report and delivery) are written after this one is done, because they reuse the builder, the runner and the training engine created here.

## Global Constraints

- Code, identifiers, docstrings and commit messages in English. Notebook prose, figure labels and report in Portuguese.
- No code comments and no inline comments. Document every function with a docstring.
- Prefer named imports (`from pathlib import Path`, `from subprocess import run`). Plain `import module` is accepted only for `sys`, `re`, `json`, `time`, `random`, `torch`, `nbformat`, `numpy as np` and `matplotlib.pyplot as plt`.
- No emojis anywhere.
- Commit messages: no co-author line, no mention of AI tools.
- Each notebook is self-contained. It imports nothing from this repository.
- Each notebook runs start to finish on Colab with a T4 GPU with no edits, and targets under 30 minutes.
- Each notebook states memory needs and estimated run time at the top.
- Local Python and torch match Colab: Python 3.13, torch 2.11.
- `src/dlv` is the only place where logic lives. Files in `notebooks/` are produced by the build and the Colab run, never edited by hand.
- Every run writes metrics to `results/` and figures to `report/figures/`.
- Tests run on CPU with tiny inputs. While iterating, run only the test file being worked on. Run the full suite once per task, before committing. Never run two test suites at the same time.
- Report prose: simple language, bullet points, short sentences. Do not use the "não é A, é B" construction.
- Never leave a Colab session running. Every Colab run ends with `colab stop`.

## Verified facts about the Colab CLI

These were tested on 2026-10-03 with a free account. Rely on them.

- `colab new -s NAME --gpu T4` creates a session with a Tesla T4 (15.6 GB), Python 3.13, torch 2.11.
- `colab exec -s NAME -f notebook.ipynb` runs every cell in `/content` and writes `notebook_output.ipynb`, with outputs, into the local current directory.
- `colab exec` exits with code 0 even when a cell raises, and it keeps running the next cells. Failures must be detected by reading the output notebook.
- `colab download -s NAME /content/file.zip local.zip` works with an absolute remote path. A bare relative path fails.
- `colab stop -s NAME` releases the machine.

## Review Focus

1. A notebook cell raises on Colab while the CLI exits 0: the runner must exit non-zero and print which cell failed. Test in Task 3.
2. The runner fails or is interrupted mid-run: the Colab session must still be stopped. Test in Task 3.
3. A notebook module uses `import dlv` or a multi-line `from dlv ... import (`: the builder must fail loudly, because the stripped notebook would be broken. Test in Task 2.
4. The dataset folder contains an unreadable image or a non-image file: it is skipped at listing time, so training never crashes mid-epoch. Test in Task 4.
5. The notebook is run twice in the same session: the dataset is not downloaded again. Test in Task 4.

## File Structure

```
pyproject.toml              dependencies and pytest settings
.python-version             3.13
Makefile                    setup, test, notebooks, colab-a3, report, zip, clean, help
.gitignore
src/dlv/__init__.py
src/dlv/data.py             Kaggle download, image listing, stratified split, dataset class
src/dlv/engine.py           training loop and evaluation
src/dlv/cnn.py              frozen ResNet-50 builder and ImageNet transform
src/dlv/figures.py          training curves, confusion matrix, per-class accuracy
scripts/build_notebooks.py  cell files + modules -> notebook
scripts/colab_run.py        run a notebook on Colab, check errors, collect outputs
scripts/build_report.py     Markdown sections -> PDF
notebooks_src/a3/           ordered cell files for the Activity 3 notebook
notebooks/                  executed notebooks (deliverable)
build/                      generated, ignored by git
results/a3.json             metrics from the Colab run
report/secoes/*.md          report sources
report/figures/a3_*.png     figures from the Colab run
tests/                      one test file per module or script
```

The spec lists `scripts/colab_run.sh`. This plan uses `scripts/colab_run.py` instead, because the runner must parse the output notebook to detect failed cells. The spec does not list `cnn.py`; it is added here to hold the ResNet-50 builder.

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`, `.python-version`, `Makefile`, `src/dlv/__init__.py`, `tests/conftest.py`, `tests/test_package.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing
- Produces: an installable package `dlv`; `uv run pytest` works; `scripts/` is on the pytest import path, so tests can `from build_notebooks import ...`

- [ ] **Step 1: Create the working branch**

```bash
git checkout -b foundation-and-activity-3
```

- [ ] **Step 2: Write `.python-version`**

```
3.13
```

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[project]
name = "dlv"
version = "0.1.0"
description = "Computer Vision course project (INFNET 26E3_3)"
requires-python = ">=3.13"
dependencies = [
    "torch~=2.11.0",
    "torchvision~=0.26.0",
    "numpy>=2.0",
    "matplotlib>=3.9",
    "scikit-learn>=1.5",
    "pillow>=10.4",
    "requests>=2.32",
]

[dependency-groups]
dev = ["pytest>=8.3", "nbformat>=5.10"]
report = ["markdown>=3.6", "xhtml2pdf>=0.2.16"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["scripts"]
```

- [ ] **Step 4: Replace `.gitignore`**

```
/req
.venv/
__pycache__/
*.pyc
*.egg-info/
.ipynb_checkpoints/
.DS_Store
.remember/
/data/
/build/
/entrega/
```

- [ ] **Step 5: Write `src/dlv/__init__.py`**

```python
"""Shared code for the Computer Vision course project."""
```

- [ ] **Step 6: Write `tests/conftest.py`**

```python
"""Shared pytest setup: headless plotting and figure cleanup."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def close_figures():
    """Closes every matplotlib figure after each test."""
    yield
    plt.close("all")
```

- [ ] **Step 7: Write `tests/test_package.py`**

```python
"""Checks that the package is installed and importable."""

import dlv


def test_package_has_docstring():
    assert "Computer Vision" in dlv.__doc__
```

- [ ] **Step 8: Write `Makefile`** (recipe lines start with a tab)

```make
.PHONY: help setup test notebooks colab-a3 report zip clean

help: ## show this help
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  make %-12s %s\n", $$1, $$2}'

setup: ## install dependencies with uv
	uv sync --all-groups

test: ## run the CPU test suite
	uv run pytest -q

notebooks: ## generate the notebooks into build/notebooks
	uv run python scripts/build_notebooks.py

colab-a3: notebooks ## run Activity 3 on a Colab T4 and collect results
	uv run python scripts/colab_run.py a3

report: ## build the PDF report
	uv run python scripts/build_report.py

zip: report ## build the delivery archive
	mkdir -p entrega
	rm -f entrega/anderson_correa_computer-vision_pd.zip
	zip -j entrega/anderson_correa_computer-vision_pd.zip notebooks/*.ipynb report/anderson_correa_deep-learning-and-vision_computer-vision.pdf

clean: ## remove generated files
	rm -rf build entrega
```

- [ ] **Step 9: Install and run the test**

Run: `make setup && uv run pytest tests/test_package.py -q`
Expected: `1 passed`

If `uv sync` cannot resolve `torch~=2.11.0` with `torchvision~=0.26.0` on this machine, stop and report the resolver message. Do not loosen the pins without asking.

- [ ] **Step 10: Commit**

```bash
git add .python-version pyproject.toml uv.lock .gitignore Makefile src/dlv/__init__.py tests/conftest.py tests/test_package.py
git commit -m "Add project scaffold with uv, make and pytest"
```

---

### Task 2: Notebook builder

**Files:**
- Create: `scripts/build_notebooks.py`
- Test: `tests/test_build_notebooks.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `NOTEBOOKS: dict[str, str]` mapping activity key to notebook name, starting with `{"a3": "A3_cnn_kaggle"}`
  - `BUILD_DIR: Path` equal to `<repo>/build`
  - `strip_package_imports(source: str) -> str`
  - `build_notebook(cells_dir: Path, package_dir: Path) -> nbformat.NotebookNode`
  - `main() -> None`, writing `build/notebooks/<name>.ipynb` for every entry in `NOTEBOOKS`

How a cell folder becomes a notebook. Files are read in filename order:

- `*.md` becomes a Markdown cell.
- `*.py` becomes a code cell, with `from dlv... import ...` lines removed.
- `*.modules` lists package module names, one per line. Each becomes a code cell holding the source of `src/dlv/<name>.py`, with `from dlv... import ...` lines removed.
- Any other file is ignored.

- [ ] **Step 1: Write the failing tests**

`tests/test_build_notebooks.py`:

```python
"""Tests for the notebook builder."""

import nbformat
import pytest

from build_notebooks import build_notebook, strip_package_imports


def make_sources(tmp_path):
    """Creates a tiny package folder and a tiny cell folder for the builder."""
    package = tmp_path / "pkg"
    cells = tmp_path / "cells"
    package.mkdir()
    cells.mkdir()
    (package / "alpha.py").write_text(
        '"""Alpha module."""\n\nfrom dlv.beta import thing\n\n\ndef alpha():\n    """Returns one."""\n    return 1\n'
    )
    (cells / "00_intro.md").write_text("# Titulo\n")
    (cells / "01_support.modules").write_text("alpha\n")
    (cells / "02_run.py").write_text("from dlv.alpha import alpha\n\nprint(alpha())\n")
    (cells / ".DS_Store").write_text("ignored")
    return cells, package


def test_strip_removes_single_line_package_imports():
    source = "from pathlib import Path\nfrom dlv.data import load\n\nx = 1\n"
    assert strip_package_imports(source) == "from pathlib import Path\n\nx = 1\n"


def test_strip_rejects_multiline_package_import():
    with pytest.raises(ValueError, match="single-line"):
        strip_package_imports("from dlv.data import (\n    load,\n)\n")


def test_strip_rejects_plain_package_import():
    with pytest.raises(ValueError, match="single-line"):
        strip_package_imports("import dlv.data\n")


def test_build_orders_cells_by_filename_and_inlines_modules(tmp_path):
    cells, package = make_sources(tmp_path)
    notebook = build_notebook(cells, package)
    assert [cell.cell_type for cell in notebook.cells] == ["markdown", "code", "code"]
    assert notebook.cells[0].source == "# Titulo"
    assert "def alpha" in notebook.cells[1].source
    assert "print(alpha())" in notebook.cells[2].source


def test_built_notebook_has_no_package_imports(tmp_path):
    cells, package = make_sources(tmp_path)
    notebook = build_notebook(cells, package)
    assert "from dlv" not in "".join(cell.source for cell in notebook.cells)


def test_built_notebook_is_valid_and_requests_t4(tmp_path):
    cells, package = make_sources(tmp_path)
    notebook = build_notebook(cells, package)
    nbformat.validate(notebook)
    assert notebook.metadata["accelerator"] == "GPU"
    assert notebook.metadata["colab"]["gpuType"] == "T4"


def test_build_fails_when_module_is_missing(tmp_path):
    cells, package = make_sources(tmp_path)
    (cells / "01_support.modules").write_text("missing\n")
    with pytest.raises(FileNotFoundError):
        build_notebook(cells, package)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_build_notebooks.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'build_notebooks'`

- [ ] **Step 3: Write `scripts/build_notebooks.py`**

```python
"""Builds self-contained notebooks from cell folders and package modules.

A cell folder holds ordered files. Markdown files become text cells, Python
files become code cells, and ``.modules`` files pull whole modules of the
``dlv`` package into the notebook. Imports from ``dlv`` are removed, because
inside the notebook every name lives in the same namespace.
"""

import re
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = ROOT / "src" / "dlv"
CELLS_DIR = ROOT / "notebooks_src"
BUILD_DIR = ROOT / "build"
NOTEBOOKS = {"a3": "A3_cnn_kaggle"}

PACKAGE_IMPORT = re.compile(r"^from dlv(\.\w+)* import [^(\n]+\n", re.MULTILINE)
UNSUPPORTED_IMPORT = re.compile(
    r"^(import dlv|from dlv(\.\w+)* import \()", re.MULTILINE
)


def strip_package_imports(source: str) -> str:
    """Removes single-line ``from dlv... import ...`` statements from source.

    Raises ValueError for import forms that cannot be removed safely.
    """
    if UNSUPPORTED_IMPORT.search(source):
        raise ValueError(
            "only single-line 'from dlv.<module> import <names>' imports are supported"
        )
    return PACKAGE_IMPORT.sub("", source).strip() + "\n"


def module_cells(listing: str, package_dir: Path) -> list:
    """Returns one code cell per module named in a ``.modules`` listing."""
    names = [line.strip() for line in listing.splitlines() if line.strip()]
    return [
        new_code_cell(strip_package_imports((package_dir / f"{name}.py").read_text()))
        for name in names
    ]


def build_notebook(cells_dir: Path, package_dir: Path) -> nbformat.NotebookNode:
    """Assembles a notebook from the files of a cell folder, in filename order."""
    cells = []
    for path in sorted(Path(cells_dir).iterdir()):
        text = path.read_text() if path.suffix in {".md", ".py", ".modules"} else ""
        if path.suffix == ".md":
            cells.append(new_markdown_cell(text.strip()))
        elif path.suffix == ".py":
            cells.append(new_code_cell(strip_package_imports(text)))
        elif path.suffix == ".modules":
            cells.extend(module_cells(text, Path(package_dir)))
    metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
    }
    return new_notebook(cells=cells, metadata=metadata)


def main() -> None:
    """Writes every notebook listed in NOTEBOOKS into ``build/notebooks``."""
    output_dir = BUILD_DIR / "notebooks"
    output_dir.mkdir(parents=True, exist_ok=True)
    for activity, name in NOTEBOOKS.items():
        notebook = build_notebook(CELLS_DIR / activity, PACKAGE_DIR)
        target = output_dir / f"{name}.ipynb"
        nbformat.write(notebook, target)
        print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_build_notebooks.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/build_notebooks.py tests/test_build_notebooks.py
git commit -m "Add notebook builder that inlines package modules"
```

---

### Task 3: Colab runner

**Files:**
- Create: `scripts/colab_run.py`
- Test: `tests/test_colab_run.py`

**Interfaces:**
- Consumes: `NOTEBOOKS`, `BUILD_DIR` from `build_notebooks`
- Produces:
  - `find_errors(notebook_path: Path) -> list[str]`
  - `collect_outputs(archive: Path, activity: str, results_dir: Path, figures_dir: Path) -> None`
  - `run_on_colab(activity: str, execute=run) -> int`, where `run` is `subprocess.run`
  - Contract for every notebook: its last cell writes `/content/outputs_<activity>.zip` containing `metrics.json` at the top level and PNG files under `figures/`.
  - After a successful run: `results/<activity>.json`, `report/figures/<activity>_<name>.png`, `notebooks/<notebook name>.ipynb` (executed, with outputs).

- [ ] **Step 1: Write the failing tests**

`tests/test_colab_run.py`:

```python
"""Tests for the Colab runner."""

import json
from subprocess import CalledProcessError
from zipfile import ZipFile

import nbformat
from nbformat.v4 import new_code_cell, new_notebook, new_output

import colab_run
from colab_run import collect_outputs, find_errors, run_on_colab


def write_notebook(path, outputs):
    """Writes a one-code-cell notebook with the given outputs."""
    cell = new_code_cell("x = 1", outputs=outputs)
    nbformat.write(new_notebook(cells=[cell]), path)


def test_find_errors_reports_failed_cells(tmp_path):
    path = tmp_path / "failed.ipynb"
    error = new_output("error", ename="RuntimeError", evalue="boom", traceback=[])
    write_notebook(path, [error])
    assert find_errors(path) == ["cell 1: RuntimeError: boom"]


def test_find_errors_is_empty_for_clean_notebook(tmp_path):
    path = tmp_path / "clean.ipynb"
    write_notebook(path, [new_output("stream", name="stdout", text="ok")])
    assert find_errors(path) == []


def test_collect_outputs_places_metrics_and_figures(tmp_path):
    archive = tmp_path / "outputs.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("metrics.json", json.dumps({"accuracy": 0.5}))
        bundle.writestr("figures/history.png", b"png-bytes")
    results = tmp_path / "results"
    figures = tmp_path / "figures"
    collect_outputs(archive, "a3", results, figures)
    assert json.loads((results / "a3.json").read_text()) == {"accuracy": 0.5}
    assert (figures / "a3_history.png").read_bytes() == b"png-bytes"


def prepare_build(tmp_path, monkeypatch):
    """Points the runner at a temporary build folder holding a notebook to run."""
    notebooks = tmp_path / "build" / "notebooks"
    notebooks.mkdir(parents=True)
    write_notebook(notebooks / "A3_cnn_kaggle.ipynb", [])
    monkeypatch.setattr(colab_run, "BUILD_DIR", tmp_path / "build")
    monkeypatch.setattr(colab_run, "ROOT", tmp_path)


def test_session_is_stopped_when_execution_fails(tmp_path, monkeypatch):
    prepare_build(tmp_path, monkeypatch)
    commands = []

    def execute(command, **kwargs):
        commands.append(command[:2])
        if command[1] == "exec":
            raise CalledProcessError(1, command)

    assert run_on_colab("a3", execute=execute) == 1
    assert commands[-1] == ["colab", "stop"]


def test_run_fails_when_a_cell_raised(tmp_path, monkeypatch):
    prepare_build(tmp_path, monkeypatch)
    commands = []

    def execute(command, **kwargs):
        commands.append(command[:2])
        if command[1] == "exec":
            error = new_output("error", ename="ValueError", evalue="bad", traceback=[])
            write_notebook(kwargs["cwd"] / "A3_cnn_kaggle_output.ipynb", [error])

    assert run_on_colab("a3", execute=execute) == 1
    assert ["colab", "download"] not in commands
    assert commands[-1] == ["colab", "stop"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_colab_run.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'colab_run'`

- [ ] **Step 3: Write `scripts/colab_run.py`**

```python
"""Runs one generated notebook on a Colab T4 session and collects its outputs.

The Colab CLI exits with code 0 even when a notebook cell raises, so the
executed notebook is inspected for error outputs. The session is always
stopped, whether the run succeeds or fails.
"""

import sys
from pathlib import Path
from shutil import copyfile
from subprocess import CalledProcessError, run
from zipfile import ZipFile

import nbformat

from build_notebooks import BUILD_DIR, NOTEBOOKS

ROOT = Path(__file__).resolve().parent.parent


def find_errors(notebook_path: Path) -> list[str]:
    """Returns one line per error output found in an executed notebook."""
    notebook = nbformat.read(notebook_path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    return [
        f"cell {number}: {output.ename}: {output.evalue}"
        for number, cell in enumerate(code_cells, start=1)
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]


def collect_outputs(
    archive: Path, activity: str, results_dir: Path, figures_dir: Path
) -> None:
    """Copies metrics and figures from a run archive into the repository."""
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if name == "metrics.json":
                (results_dir / f"{activity}.json").write_bytes(bundle.read(name))
            elif name.startswith("figures/") and name.endswith(".png"):
                target = figures_dir / f"{activity}_{Path(name).name}"
                target.write_bytes(bundle.read(name))


def run_on_colab(activity: str, execute=run) -> int:
    """Executes the notebook of an activity on Colab and returns an exit code."""
    name = NOTEBOOKS[activity]
    session = f"dlv-{activity}"
    work_dir = BUILD_DIR / "runs" / activity
    work_dir.mkdir(parents=True, exist_ok=True)
    copyfile(BUILD_DIR / "notebooks" / f"{name}.ipynb", work_dir / f"{name}.ipynb")
    executed = work_dir / f"{name}_output.ipynb"
    archive = work_dir / "outputs.zip"
    executed.unlink(missing_ok=True)
    archive.unlink(missing_ok=True)
    try:
        execute(["colab", "stop", "-s", session], check=False, capture_output=True)
        execute(["colab", "new", "-s", session, "--gpu", "T4"], check=True)
        execute(
            ["colab", "exec", "-s", session, "-f", f"{name}.ipynb"],
            check=True,
            cwd=work_dir,
        )
        errors = find_errors(executed)
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
        execute(
            ["colab", "download", "-s", session, f"/content/outputs_{activity}.zip", str(archive)],
            check=True,
        )
    except (CalledProcessError, FileNotFoundError) as failure:
        print(f"colab run failed: {failure}", file=sys.stderr)
        return 1
    finally:
        execute(["colab", "stop", "-s", session], check=False)
    collect_outputs(archive, activity, ROOT / "results", ROOT / "report" / "figures")
    (ROOT / "notebooks").mkdir(exist_ok=True)
    copyfile(executed, ROOT / "notebooks" / f"{name}.ipynb")
    return 0


if __name__ == "__main__":
    sys.exit(run_on_colab(sys.argv[1]))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_colab_run.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/colab_run.py tests/test_colab_run.py
git commit -m "Add Colab runner that detects failed cells and always stops the session"
```

---

### Task 4: Data module

**Files:**
- Create: `src/dlv/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `download_kaggle_dataset(slug: str, destination: Path) -> Path`
  - `list_image_folder(root: Path) -> tuple[list[Path], list[int], list[str]]` returning paths, integer labels and sorted class names. The class of an image is the name of its parent folder.
  - `stratified_split(labels: list[int], fractions: tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 42) -> tuple[list[int], list[int], list[int]]`
  - `ImageListDataset(paths, labels, transform)`, a `torch.utils.data.Dataset` yielding `(tensor, int)`

- [ ] **Step 1: Write the failing tests**

`tests/test_data.py`:

```python
"""Tests for dataset download, listing and splitting."""

from collections import Counter
from io import BytesIO
from zipfile import ZipFile

import pytest
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from dlv import data
from dlv.data import (
    ImageListDataset,
    download_kaggle_dataset,
    list_image_folder,
    stratified_split,
)


def make_image_tree(root, classes=("bike", "car"), per_class=10):
    """Creates a folder per class with small solid-colour images."""
    for name in classes:
        folder = root / "data" / name
        folder.mkdir(parents=True)
        for index in range(per_class):
            Image.new("RGB", (8, 8), (index * 20, 0, 0)).save(folder / f"{name}_{index}.bmp")
    return root


class FakeResponse:
    """Stands in for a streamed HTTP response carrying a zip archive."""

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *details):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield self.payload


def zip_payload():
    """Returns the bytes of a zip archive holding one tiny image."""
    image = BytesIO()
    Image.new("RGB", (4, 4)).save(image, format="PNG")
    buffer = BytesIO()
    with ZipFile(buffer, "w") as bundle:
        bundle.writestr("data/bike/one.png", image.getvalue())
    return buffer.getvalue()


def test_download_extracts_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "get", lambda *args, **kwargs: FakeResponse(zip_payload()))
    destination = download_kaggle_dataset("owner/name", tmp_path / "set")
    assert (destination / "data" / "bike" / "one.png").exists()
    assert not (destination / "archive.zip").exists()


def test_download_is_skipped_when_already_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "get", lambda *args, **kwargs: FakeResponse(zip_payload()))
    download_kaggle_dataset("owner/name", tmp_path / "set")

    def fail(*args, **kwargs):
        raise AssertionError("downloaded twice")

    monkeypatch.setattr(data, "get", fail)
    assert download_kaggle_dataset("owner/name", tmp_path / "set") == tmp_path / "set"


def test_list_image_folder_labels_by_parent_folder(tmp_path):
    make_image_tree(tmp_path)
    paths, labels, class_names = list_image_folder(tmp_path)
    assert class_names == ["bike", "car"]
    assert len(paths) == 20
    assert Counter(labels) == {0: 10, 1: 10}
    assert all(class_names[label] == path.parent.name for path, label in zip(paths, labels))


def test_list_image_folder_skips_unreadable_and_non_image_files(tmp_path):
    make_image_tree(tmp_path)
    (tmp_path / "data" / "bike" / "broken.jpg").write_bytes(b"not an image")
    (tmp_path / "data" / "bike" / "notes.txt").write_text("hello")
    (tmp_path / "data" / "bike" / ".hidden.png").write_bytes(b"x")
    paths, _, _ = list_image_folder(tmp_path)
    assert len(paths) == 20
    assert all(path.suffix == ".bmp" for path in paths)


def test_stratified_split_keeps_proportions_and_is_disjoint():
    labels = [0] * 100 + [1] * 40
    train, val, test = stratified_split(labels)
    assert sorted(train + val + test) == list(range(140))
    assert Counter(labels[i] for i in train) == {0: 70, 1: 28}
    assert Counter(labels[i] for i in val) == {0: 15, 1: 6}
    assert Counter(labels[i] for i in test) == {0: 15, 1: 6}


def test_stratified_split_is_deterministic():
    labels = [0] * 50 + [1] * 50
    assert stratified_split(labels, seed=7) == stratified_split(labels, seed=7)


def test_stratified_split_names_the_class_that_is_too_small():
    with pytest.raises(ValueError, match="class 1 has 3 images"):
        stratified_split([0] * 50 + [1] * 3)


def test_image_list_dataset_returns_tensor_and_label(tmp_path):
    make_image_tree(tmp_path)
    paths, labels, _ = list_image_folder(tmp_path)
    dataset = ImageListDataset(paths, labels, ToTensor())
    image, label = dataset[0]
    assert len(dataset) == 20
    assert image.shape == (3, 8, 8)
    assert image.dtype == torch.float32
    assert label == labels[0]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_data.py -q`
Expected: collection error, `cannot import name 'data' from 'dlv'`

- [ ] **Step 3: Write `src/dlv/data.py`**

```python
"""Dataset download, image discovery and splitting shared by the notebooks."""

from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
from requests import get
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

KAGGLE_DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/{slug}"
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png"}
MINIMUM_IMAGES_PER_CLASS = 7


def download_kaggle_dataset(slug: str, destination: Path) -> Path:
    """Downloads and extracts a public Kaggle dataset, once.

    A marker file records a finished download, so calling this again for the
    same destination returns immediately. No Kaggle account is needed.
    """
    destination = Path(destination)
    marker = destination / ".complete"
    if marker.exists():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "archive.zip"
    with get(KAGGLE_DOWNLOAD_URL.format(slug=slug), stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(archive, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    with ZipFile(archive) as bundle:
        bundle.extractall(destination)
    archive.unlink()
    marker.touch()
    return destination


def is_readable_image(path: Path) -> bool:
    """Tells whether a file is a visible image that Pillow can decode."""
    if path.name.startswith(".") or path.suffix.lower() not in IMAGE_SUFFIXES:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception:
        return False
    return True


def list_image_folder(root: Path) -> tuple[list[Path], list[int], list[str]]:
    """Finds every readable image under root and labels it by its parent folder.

    Returns the image paths, one integer label per path, and the class names
    in the order used by the labels. Unreadable files are left out.
    """
    paths = sorted(path for path in Path(root).rglob("*") if path.is_file() and is_readable_image(path))
    class_names = sorted({path.parent.name for path in paths})
    index = {name: position for position, name in enumerate(class_names)}
    labels = [index[path.parent.name] for path in paths]
    return paths, labels, class_names


def stratified_split(
    labels: list[int],
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Splits indices into train, validation and test, keeping class shares.

    Raises ValueError naming the class when a class is too small to appear in
    all three parts.
    """
    for label, count in sorted(Counter(labels).items()):
        if count < MINIMUM_IMAGES_PER_CLASS:
            raise ValueError(
                f"class {label} has {count} images; at least {MINIMUM_IMAGES_PER_CLASS} are needed to split"
            )
    indices = list(range(len(labels)))
    train, rest = train_test_split(
        indices, train_size=fractions[0], stratify=labels, random_state=seed
    )
    validation_share = fractions[1] / (fractions[1] + fractions[2])
    validation, test = train_test_split(
        rest,
        train_size=validation_share,
        stratify=[labels[i] for i in rest],
        random_state=seed,
    )
    return sorted(train), sorted(validation), sorted(test)


class ImageListDataset(Dataset):
    """Serves transformed images and labels from parallel lists."""

    def __init__(self, paths: list[Path], labels: list[int], transform):
        """Stores the image paths, their labels and the transform to apply."""
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """Returns the number of images."""
        return len(self.paths)

    def __getitem__(self, index: int):
        """Loads one image as RGB and returns it transformed, with its label."""
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB")), self.labels[index]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_data.py -q`
Expected: `8 passed`

If the proportion test fails by one image in a class, the rounding of scikit-learn differs from the expected counts. Report the actual counts instead of changing the test.

- [ ] **Step 5: Commit**

```bash
git add src/dlv/data.py tests/test_data.py
git commit -m "Add dataset download, image listing and stratified split"
```

---

### Task 5: Training engine

**Files:**
- Create: `src/dlv/engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `train_all(model: nn.Module) -> None`, the default train-mode setter
  - `run_epoch(model, loader, device, optimizer=None) -> tuple[float, float]` returning mean loss and accuracy; trains when an optimizer is given
  - `fit(model, train_loader, validation_loader, optimizer, epochs: int, device, set_train_mode=train_all) -> dict[str, list[float]]` with keys `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`
  - `evaluate(model, loader, device, num_classes: int) -> dict` with keys `accuracy` (float), `per_class_accuracy` (list of float), `confusion` (nested list, rows are true classes)

- [ ] **Step 1: Write the failing tests**

`tests/test_engine.py`:

```python
"""Tests for the training loop and evaluation."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dlv.engine import evaluate, fit, run_epoch


def separable_loader():
    """Returns a loader over two well separated two-dimensional classes."""
    torch.manual_seed(0)
    features = torch.cat([torch.randn(40, 2) + 4, torch.randn(40, 2) - 4])
    labels = torch.cat([torch.zeros(40, dtype=torch.long), torch.ones(40, dtype=torch.long)])
    return DataLoader(TensorDataset(features, labels), batch_size=16, shuffle=True)


class AlwaysFirstClass(nn.Module):
    """Predicts class zero for every input."""

    def forward(self, inputs):
        """Returns logits that favour class zero."""
        logits = torch.zeros(inputs.size(0), 2)
        logits[:, 0] = 1.0
        return logits


def test_fit_reduces_loss_and_records_every_epoch():
    torch.manual_seed(0)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = separable_loader()
    history = fit(model, loader, loader, optimizer, epochs=5, device=torch.device("cpu"))
    assert set(history) == {"train_loss", "train_accuracy", "val_loss", "val_accuracy"}
    assert all(len(values) == 5 for values in history.values())
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert history["val_accuracy"][-1] > 0.95


def test_fit_calls_train_mode_setter_each_epoch():
    calls = []
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = separable_loader()
    fit(
        model,
        loader,
        loader,
        optimizer,
        epochs=3,
        device=torch.device("cpu"),
        set_train_mode=lambda module: calls.append(module),
    )
    assert calls == [model, model, model]


def test_run_epoch_without_optimizer_leaves_weights_untouched():
    model = nn.Linear(2, 2)
    before = model.weight.clone()
    run_epoch(model, separable_loader(), torch.device("cpu"))
    assert torch.equal(model.weight, before)


def test_run_epoch_on_empty_loader_returns_zeros():
    empty = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))
    assert run_epoch(nn.Linear(2, 2), empty, torch.device("cpu")) == (0.0, 0.0)


def test_evaluate_reports_per_class_accuracy_and_confusion():
    loader = separable_loader()
    report = evaluate(AlwaysFirstClass(), loader, torch.device("cpu"), num_classes=2)
    assert report["accuracy"] == 0.5
    assert report["per_class_accuracy"] == [1.0, 0.0]
    assert report["confusion"] == [[40, 0], [40, 0]]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_engine.py -q`
Expected: collection error, `No module named 'dlv.engine'`

- [ ] **Step 3: Write `src/dlv/engine.py`**

```python
"""Training loop and evaluation for image classifiers."""

import torch
from torch import nn
from torch.nn.functional import cross_entropy


def train_all(model: nn.Module) -> None:
    """Puts every layer of the model in training mode."""
    model.train()


def run_epoch(model, loader, device, optimizer=None) -> tuple[float, float]:
    """Runs one pass over a loader and returns mean loss and accuracy.

    The model is updated only when an optimizer is given. An empty loader
    yields zeros.
    """
    training = optimizer is not None
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = cross_entropy(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += labels.size(0)
    return total_loss / max(count, 1), correct / max(count, 1)


def fit(
    model,
    train_loader,
    validation_loader,
    optimizer,
    epochs: int,
    device,
    set_train_mode=train_all,
) -> dict[str, list[float]]:
    """Trains a classifier and returns loss and accuracy for every epoch.

    ``set_train_mode`` decides which layers train. Pass a custom function to
    keep a frozen backbone in evaluation mode.
    """
    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    for epoch in range(epochs):
        set_train_mode(model)
        train_loss, train_accuracy = run_epoch(model, train_loader, device, optimizer)
        model.eval()
        val_loss, val_accuracy = run_epoch(model, validation_loader, device)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        print(
            f"epoch {epoch + 1}/{epochs} "
            f"train loss {train_loss:.4f} acc {train_accuracy:.4f} "
            f"val loss {val_loss:.4f} acc {val_accuracy:.4f}"
        )
    return history


def evaluate(model, loader, device, num_classes: int) -> dict:
    """Measures overall accuracy, accuracy per class and the confusion matrix.

    Confusion rows are true classes and columns are predicted classes.
    """
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu()
            for truth, prediction in zip(labels.tolist(), predictions.tolist()):
                confusion[truth, prediction] += 1
    hits = confusion.diag()
    per_class = (hits / confusion.sum(dim=1).clamp(min=1)).tolist()
    accuracy = (hits.sum() / confusion.sum().clamp(min=1)).item()
    return {
        "accuracy": accuracy,
        "per_class_accuracy": per_class,
        "confusion": confusion.tolist(),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_engine.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/dlv/engine.py tests/test_engine.py
git commit -m "Add training loop and per-class evaluation"
```

---

### Task 6: Frozen ResNet-50

**Files:**
- Create: `src/dlv/cnn.py`
- Test: `tests/test_cnn.py`

**Interfaces:**
- Consumes: `fit` from `dlv.engine` (in one test)
- Produces:
  - `IMAGENET_MEAN`, `IMAGENET_STD`
  - `build_feature_extractor(num_classes: int, pretrained: bool = True) -> nn.Module`
  - `set_feature_extraction_mode(model: nn.Module) -> None`
  - `trainable_parameters(model: nn.Module) -> list`
  - `imagenet_transform(image_size: int = 224)`

Why `set_feature_extraction_mode` exists: `model.train()` would make the batch normalization layers of the frozen backbone update their running statistics, which changes the backbone even though its weights are frozen. This function keeps the backbone in evaluation mode and only the new layer in training mode.

- [ ] **Step 1: Write the failing tests**

`tests/test_cnn.py`:

```python
"""Tests for the frozen ResNet-50 feature extractor."""

import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from dlv.cnn import (
    build_feature_extractor,
    imagenet_transform,
    set_feature_extraction_mode,
    trainable_parameters,
)
from dlv.engine import fit


def test_only_the_new_head_is_trainable():
    model = build_feature_extractor(num_classes=5, pretrained=False)
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert names == {"fc.weight", "fc.bias"}
    assert len(trainable_parameters(model)) == 2


def test_head_matches_number_of_classes():
    model = build_feature_extractor(num_classes=5, pretrained=False)
    model.eval()
    assert model.fc.out_features == 5
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 5)


def test_feature_extraction_mode_keeps_backbone_in_eval():
    model = build_feature_extractor(num_classes=5, pretrained=False)
    set_feature_extraction_mode(model)
    assert model.layer1.training is False
    assert model.bn1.training is False
    assert model.fc.training is True


def test_training_changes_head_and_leaves_backbone_untouched():
    torch.manual_seed(0)
    model = build_feature_extractor(num_classes=2, pretrained=False)
    images = torch.randn(4, 3, 64, 64)
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    optimizer = torch.optim.Adam(trainable_parameters(model), lr=0.01)
    running_mean = model.bn1.running_mean.clone()
    first_filter = model.conv1.weight.clone()
    head = model.fc.weight.clone()
    fit(
        model,
        loader,
        loader,
        optimizer,
        epochs=1,
        device=torch.device("cpu"),
        set_train_mode=set_feature_extraction_mode,
    )
    assert torch.equal(model.bn1.running_mean, running_mean)
    assert torch.equal(model.conv1.weight, first_filter)
    assert not torch.equal(model.fc.weight, head)


def test_imagenet_transform_produces_normalized_square_tensor():
    tensor = imagenet_transform()(Image.new("RGB", (640, 480), (255, 255, 255)))
    assert tensor.shape == (3, 224, 224)
    assert tensor.max() > 2.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cnn.py -q`
Expected: collection error, `No module named 'dlv.cnn'`

- [ ] **Step 3: Write `src/dlv/cnn.py`**

```python
"""Transfer learning by feature extraction with a pre-trained ResNet-50."""

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_feature_extractor(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Builds a ResNet-50 whose backbone is frozen and whose head is new.

    Every backbone weight stops receiving gradients. The classification head
    is replaced by a fresh linear layer with one output per class, which is
    the only trainable part.
    """
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def set_feature_extraction_mode(model: nn.Module) -> None:
    """Keeps the frozen backbone in evaluation mode while the head trains.

    Batch normalization layers update their running statistics in training
    mode even when their weights are frozen, so the backbone must stay in
    evaluation mode to remain a fixed feature extractor.
    """
    model.eval()
    model.fc.train()


def trainable_parameters(model: nn.Module) -> list:
    """Returns the parameters that receive gradients."""
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def imagenet_transform(image_size: int = 224) -> Compose:
    """Returns the preprocessing used when the backbone was pre-trained."""
    return Compose(
        [
            Resize(256),
            CenterCrop(image_size),
            ToTensor(),
            Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cnn.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/dlv/cnn.py tests/test_cnn.py
git commit -m "Add frozen ResNet-50 feature extractor"
```

---

### Task 7: Figures

**Files:**
- Create: `src/dlv/figures.py`
- Test: `tests/test_figures.py`

**Interfaces:**
- Consumes: the `history` dict from `fit` and the `confusion` and `per_class_accuracy` values from `evaluate`
- Produces, each saving a PNG and returning the matplotlib `Figure`:
  - `plot_history(history: dict, path: Path) -> Figure`
  - `plot_confusion(confusion: list[list[int]], class_names: list[str], path: Path) -> Figure`
  - `plot_per_class_accuracy(per_class_accuracy: list[float], class_names: list[str], path: Path) -> Figure`

Figure labels are in Portuguese, because the figures go into the report.

- [ ] **Step 1: Write the failing tests**

`tests/test_figures.py`:

```python
"""Tests for the plotting helpers."""

from dlv.figures import plot_confusion, plot_history, plot_per_class_accuracy

HISTORY = {
    "train_loss": [1.0, 0.5],
    "train_accuracy": [0.5, 0.8],
    "val_loss": [1.1, 0.7],
    "val_accuracy": [0.4, 0.7],
}


def test_plot_history_saves_loss_and_accuracy_panels(tmp_path):
    figure = plot_history(HISTORY, tmp_path / "history.png")
    assert (tmp_path / "history.png").stat().st_size > 0
    assert [axis.get_ylabel() for axis in figure.axes] == ["Perda", "Acurácia"]


def test_plot_confusion_labels_axes_with_class_names(tmp_path):
    figure = plot_confusion([[3, 1], [0, 4]], ["bike", "car"], tmp_path / "confusion.png")
    axis = figure.axes[0]
    assert (tmp_path / "confusion.png").stat().st_size > 0
    assert [label.get_text() for label in axis.get_xticklabels()] == ["bike", "car"]
    assert axis.get_ylabel() == "Classe real"


def test_plot_per_class_accuracy_draws_one_bar_per_class(tmp_path):
    figure = plot_per_class_accuracy([0.75, 1.0], ["bike", "car"], tmp_path / "per_class.png")
    assert (tmp_path / "per_class.png").stat().st_size > 0
    assert len(figure.axes[0].patches) == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_figures.py -q`
Expected: collection error, `No module named 'dlv.figures'`

- [ ] **Step 3: Write `src/dlv/figures.py`**

```python
"""Plots used by the notebooks and the report. Labels are in Portuguese."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_history(history: dict, path: Path) -> Figure:
    """Draws loss and accuracy per epoch for training and validation."""
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(11, 4))
    loss_axis.plot(epochs, history["train_loss"], marker="o", label="Treino")
    loss_axis.plot(epochs, history["val_loss"], marker="o", label="Validação")
    loss_axis.set_xlabel("Época")
    loss_axis.set_ylabel("Perda")
    loss_axis.legend()
    accuracy_axis.plot(epochs, history["train_accuracy"], marker="o", label="Treino")
    accuracy_axis.plot(epochs, history["val_accuracy"], marker="o", label="Validação")
    accuracy_axis.set_xlabel("Época")
    accuracy_axis.set_ylabel("Acurácia")
    accuracy_axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    return figure


def plot_confusion(confusion: list[list[int]], class_names: list[str], path: Path) -> Figure:
    """Draws a confusion matrix with true classes on rows."""
    size = max(5, 0.7 * len(class_names))
    figure, axis = plt.subplots(figsize=(size, size))
    axis.imshow(confusion, cmap="Blues")
    positions = range(len(class_names))
    axis.set_xticks(positions, labels=class_names, rotation=45, ha="right")
    axis.set_yticks(positions, labels=class_names)
    axis.set_xlabel("Classe prevista")
    axis.set_ylabel("Classe real")
    limit = max(max(row) for row in confusion) / 2
    for row_index, row in enumerate(confusion):
        for column_index, value in enumerate(row):
            colour = "white" if value > limit else "black"
            axis.text(column_index, row_index, str(value), ha="center", va="center", color=colour)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    return figure


def plot_per_class_accuracy(
    per_class_accuracy: list[float], class_names: list[str], path: Path
) -> Figure:
    """Draws one bar per class with its accuracy on the test set."""
    figure, axis = plt.subplots(figsize=(max(6, 0.8 * len(class_names)), 4))
    axis.bar(class_names, per_class_accuracy)
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("Classe")
    axis.set_ylabel("Acurácia")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    return figure
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_figures.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/dlv/figures.py tests/test_figures.py
git commit -m "Add training curve, confusion matrix and per-class accuracy plots"
```

---

### Task 8: Activity 3 notebook, first run on Colab

**Files:**
- Create: `notebooks_src/a3/00_header.md`, `01_setup.py`, `02_support.md`, `03_support.modules`, `04_data.md`, `05_data.py`, `06_model.md`, `07_model.py`, `08_train.md`, `09_train.py`, `10_evaluation.md`, `11_evaluation.py`, `12_archive.py`
- Test: `tests/test_a3_notebook.py`
- Produced by the run: `notebooks/A3_cnn_kaggle.ipynb`, `results/a3.json`, `report/figures/a3_history.png`, `report/figures/a3_confusion.png`, `report/figures/a3_per_class.png`

**Interfaces:**
- Consumes: `download_kaggle_dataset`, `list_image_folder`, `stratified_split`, `ImageListDataset` (Task 4); `fit`, `evaluate` (Task 5); `build_feature_extractor`, `set_feature_extraction_mode`, `trainable_parameters`, `imagenet_transform` (Task 6); `plot_history`, `plot_confusion`, `plot_per_class_accuracy` (Task 7); `build_notebook` (Task 2); `run_on_colab` (Task 3)
- Produces: `results/a3.json` with keys `class_names`, `class_counts`, `split_sizes`, `hyperparameters`, `total_parameters`, `trainable_parameters`, `history`, `accuracy`, `per_class_accuracy`, `confusion`, `runtime_seconds`, `peak_gpu_memory_mb`

- [ ] **Step 1: Write the failing test**

`tests/test_a3_notebook.py`:

```python
"""Checks that the Activity 3 notebook builds into a self-contained notebook."""

from build_notebooks import CELLS_DIR, PACKAGE_DIR, build_notebook


def test_a3_notebook_is_self_contained():
    notebook = build_notebook(CELLS_DIR / "a3", PACKAGE_DIR)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert "from dlv" not in code
    assert "import dlv" not in code
    for name in ["def download_kaggle_dataset", "def fit", "def build_feature_extractor", "def plot_history"]:
        assert name in code
    compile(code, "A3_cnn_kaggle", "exec")


def test_a3_notebook_starts_with_requirements_and_ends_with_archive():
    notebook = build_notebook(CELLS_DIR / "a3", PACKAGE_DIR)
    assert notebook.cells[0].cell_type == "markdown"
    assert "Tempo estimado" in notebook.cells[0].source
    assert "Memória" in notebook.cells[0].source
    assert "outputs_a3" in notebook.cells[-1].source
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_a3_notebook.py -q`
Expected: FAIL with `FileNotFoundError` for `notebooks_src/a3`

- [ ] **Step 3: Write `notebooks_src/a3/00_header.md`**

```markdown
# Atividade 3: Classificador com CNN pré-treinada

**Aluno:** Anderson Corrêa

**Objetivo:** classificar fotos de objetos usando transfer learning por feature extraction. A ResNet-50 pré-treinada na ImageNet fica congelada e só a última camada é treinada.

**Requisitos de execução**

- Ambiente: Google Colab com GPU T4
- Memória de GPU: menos de 4 GB
- Memória RAM: menos de 4 GB
- Disco: 2 GB livres
- Tempo estimado: menos de 15 minutos

**Como executar:** menu *Ambiente de execução*, *Executar tudo*. O notebook baixa os dados sozinho e não precisa de login.
```

- [ ] **Step 4: Write `notebooks_src/a3/01_setup.py`**

```python
import json
import random
import time
from pathlib import Path
from shutil import make_archive

import numpy as np
import torch
from torch.utils.data import DataLoader

SEED = 42
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
DATASET_SLUG = "pavansanagapati/images-dataset"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data/images_dataset")
OUTPUT_DIR = Path("outputs/a3")
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
STARTED_AT = time.time()

print("dispositivo:", torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU")
```

- [ ] **Step 5: Write `notebooks_src/a3/02_support.md` and `03_support.modules`**

`02_support.md`:

```markdown
## Código de apoio

As quatro células abaixo definem as funções usadas no restante do notebook: dados, laço de treino, modelo e gráficos.
```

`03_support.modules`:

```
data
engine
cnn
figures
```

- [ ] **Step 6: Write `notebooks_src/a3/04_data.md` and `05_data.py`**

`04_data.md`:

```markdown
## Dados

- Download do dataset público do Kaggle
- A classe de cada imagem é o nome da pasta onde ela está
- Divisão estratificada: 70% treino, 15% validação, 15% teste
- Pré-processamento igual ao do pré-treinamento: redimensionar, recorte central de 224x224 e normalização com média e desvio padrão da ImageNet
```

`05_data.py`:

```python
from collections import Counter

from dlv.cnn import imagenet_transform
from dlv.data import ImageListDataset, download_kaggle_dataset, list_image_folder, stratified_split

dataset_root = download_kaggle_dataset(DATASET_SLUG, DATA_DIR)
paths, labels, class_names = list_image_folder(dataset_root)
train_indices, validation_indices, test_indices = stratified_split(labels, seed=SEED)
transform = imagenet_transform()


def make_loader(indices, shuffle):
    """Builds a loader over the images selected by the given indices."""
    dataset = ImageListDataset(
        [paths[i] for i in indices], [labels[i] for i in indices], transform
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=2)


train_loader = make_loader(train_indices, shuffle=True)
validation_loader = make_loader(validation_indices, shuffle=False)
test_loader = make_loader(test_indices, shuffle=False)

class_counts = {class_names[label]: count for label, count in sorted(Counter(labels).items())}
split_sizes = {
    "train": len(train_indices),
    "validation": len(validation_indices),
    "test": len(test_indices),
}
print("classes:", class_names)
print("imagens por classe:", class_counts)
print("divisão:", split_sizes)
```

- [ ] **Step 7: Write `notebooks_src/a3/06_model.md` and `07_model.py`**

`06_model.md`:

```markdown
## Modelo

- ResNet-50 com pesos da ImageNet
- Todos os pesos do backbone congelados
- A camada final `fc` é trocada por uma nova, com uma saída por classe do dataset
- Só a nova camada recebe gradiente
```

`07_model.py`:

```python
from dlv.cnn import build_feature_extractor, trainable_parameters

model = build_feature_extractor(num_classes=len(class_names)).to(DEVICE)
total_parameters = sum(parameter.numel() for parameter in model.parameters())
trainable_count = sum(parameter.numel() for parameter in trainable_parameters(model))
print("parâmetros totais:", total_parameters)
print("parâmetros treináveis:", trainable_count)
print("camada final:", model.fc)
```

- [ ] **Step 8: Write `notebooks_src/a3/08_train.md` and `09_train.py`**

`08_train.md`:

```markdown
## Treino

- Um único treino da camada final
- Otimizador Adam, taxa de aprendizado 0,001, 10 épocas, lotes de 32 imagens
- O backbone fica em modo de avaliação, para que as estatísticas de batch normalization não mudem
```

`09_train.py`:

```python
from dlv.cnn import set_feature_extraction_mode, trainable_parameters
from dlv.engine import fit

optimizer = torch.optim.Adam(trainable_parameters(model), lr=LEARNING_RATE)
history = fit(
    model,
    train_loader,
    validation_loader,
    optimizer,
    epochs=EPOCHS,
    device=DEVICE,
    set_train_mode=set_feature_extraction_mode,
)
```

- [ ] **Step 9: Write `notebooks_src/a3/10_evaluation.md` and `11_evaluation.py`**

`10_evaluation.md`:

```markdown
## Avaliação no conjunto de teste

- Acurácia global
- Acurácia por classe
- Matriz de confusão
- Curvas de perda e acurácia por época
```

`11_evaluation.py`:

```python
from dlv.engine import evaluate
from dlv.figures import plot_confusion, plot_history, plot_per_class_accuracy

report = evaluate(model, test_loader, DEVICE, num_classes=len(class_names))
print(f"acurácia global: {report['accuracy']:.4f}")
for name, value in zip(class_names, report["per_class_accuracy"]):
    print(f"{name}: {value:.4f}")

plot_history(history, FIGURES_DIR / "history.png")
plot_confusion(report["confusion"], class_names, FIGURES_DIR / "confusion.png")
plot_per_class_accuracy(report["per_class_accuracy"], class_names, FIGURES_DIR / "per_class.png")
```

- [ ] **Step 10: Write `notebooks_src/a3/12_archive.py`**

```python
peak_memory = torch.cuda.max_memory_allocated() / 1e6 if DEVICE.type == "cuda" else 0.0
metrics = {
    "class_names": class_names,
    "class_counts": class_counts,
    "split_sizes": split_sizes,
    "hyperparameters": {
        "backbone": "resnet50 IMAGENET1K_V2",
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "seed": SEED,
    },
    "total_parameters": total_parameters,
    "trainable_parameters": trainable_count,
    "history": history,
    "accuracy": report["accuracy"],
    "per_class_accuracy": dict(zip(class_names, report["per_class_accuracy"])),
    "confusion": report["confusion"],
    "runtime_seconds": round(time.time() - STARTED_AT, 1),
    "peak_gpu_memory_mb": round(peak_memory, 1),
}
(OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(make_archive("outputs_a3", "zip", OUTPUT_DIR))
print("tempo total (s):", metrics["runtime_seconds"])
print("pico de memória de GPU (MB):", metrics["peak_gpu_memory_mb"])
```

- [ ] **Step 11: Run the notebook tests**

Run: `uv run pytest tests/test_a3_notebook.py -q`
Expected: `2 passed`

- [ ] **Step 12: Run the full suite once, after checking machine load**

Run: `uptime && make test`
Expected: load is not saturated; `36 passed` (1 + 7 + 5 + 8 + 5 + 5 + 3 + 2). If load is high, wait instead of running.

- [ ] **Step 13: Run on Colab**

Run: `make colab-a3`
Expected, in order:
- `wrote build/notebooks/A3_cnn_kaggle.ipynb`
- `[colab] Session READY`
- ten `epoch N/10` lines with validation accuracy rising
- `[colab] Session terminated.`
- exit code 0

If the run exits with code 1, the failing cell is printed as `cell N: ErrorName: message`. Fix the cause in `src/dlv` or `notebooks_src/a3` and run again. Do not edit files in `notebooks/` or `build/`.

- [ ] **Step 14: Verify what came back**

Run:

```bash
colab sessions
uv run python -c "
import json
import nbformat
metrics = json.load(open('results/a3.json'))
print('accuracy', metrics['accuracy'])
print('classes', metrics['class_names'])
print('split', metrics['split_sizes'])
print('trainable', metrics['trainable_parameters'], 'of', metrics['total_parameters'])
print('runtime', metrics['runtime_seconds'], 'peak MB', metrics['peak_gpu_memory_mb'])
notebook = nbformat.read('notebooks/A3_cnn_kaggle.ipynb', as_version=4)
images = sum('image/png' in output.get('data', {}) for cell in notebook.cells if cell.cell_type == 'code' for output in cell.outputs)
print('inline images', images)
"
ls -la report/figures/
```

Expected:
- `No active sessions found on server`
- accuracy above 0.85 (ImageNet features on everyday objects; a much lower value means a bug, most likely in labels or normalization)
- split sizes add up to the number of images, about 1,800
- trainable parameters equal `2048 * number_of_classes + number_of_classes`
- at least 3 inline images in the executed notebook
- `a3_history.png`, `a3_confusion.png`, `a3_per_class.png` present and non-empty

If inline images is 0, the executed notebook has no figures. Add `plt.show()` after the three plot calls in `11_evaluation.py` (with `import matplotlib.pyplot as plt` at the top of that cell file), then run `make colab-a3` again.

- [ ] **Step 15: Commit**

```bash
git add notebooks_src/a3 tests/test_a3_notebook.py notebooks/A3_cnn_kaggle.ipynb results/a3.json report/figures/a3_history.png report/figures/a3_confusion.png report/figures/a3_per_class.png
git commit -m "Add Activity 3 notebook with first Colab T4 run"
```

---

### Task 9: Report builder, Activity 3 report section, final run

**Files:**
- Create: `scripts/build_report.py`, `report/secoes/00_identificacao.md`, `report/secoes/30_atividade3.md`, `notebooks_src/a3/13_analysis.md`
- Modify: `notebooks_src/a3/00_header.md` (measured memory and time), `notebooks_src/a3/12_archive.py` is renamed to `14_archive.py` so the archive cell stays last
- Test: `tests/test_build_report.py`

**Interfaces:**
- Consumes: `results/a3.json` and `report/figures/a3_*.png` from Task 8
- Produces:
  - `merge_sections(sections_dir: Path) -> str`
  - `build(sections_dir: Path, output_pdf: Path) -> None`
  - `report/anderson_correa_deep-learning-and-vision_computer-vision.pdf`
  - Section file naming for later plans: `10_atividade1.md`, `20_atividade2.md`, `30_atividade3.md`, `40_atividade4_raio_x.md`, `45_atividade4_trafego.md`, `90_uso_de_ia_e_referencias.md`

- [ ] **Step 1: Write the failing tests**

`tests/test_build_report.py`:

```python
"""Tests for the report builder."""

from build_report import build, merge_sections


def test_merge_sections_follows_filename_order(tmp_path):
    (tmp_path / "30_c.md").write_text("# Terceira\n")
    (tmp_path / "00_a.md").write_text("# Primeira\n")
    merged = merge_sections(tmp_path)
    assert merged.index("Primeira") < merged.index("Terceira")


def test_build_writes_pdf_with_accents_and_table(tmp_path):
    (tmp_path / "00_a.md").write_text(
        "# Relatório\n\n- Acurácia de validação\n\n| Classe | Acurácia |\n| --- | --- |\n| bike | 0,98 |\n"
    )
    output = tmp_path / "out.pdf"
    build(tmp_path, output)
    assert output.read_bytes().startswith(b"%PDF")
    assert output.stat().st_size > 1000
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_build_report.py -q`
Expected: collection error, `No module named 'build_report'`

- [ ] **Step 3: Write `scripts/build_report.py`**

```python
"""Joins the report sections and renders the PDF deliverable.

Sections live in ``report/secoes`` and are joined in filename order, so the
numeric prefixes define the structure of the document. The toolchain is pure
Python: ``markdown`` for HTML and ``xhtml2pdf`` for the PDF.
"""

import re
from pathlib import Path

from markdown import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "report"
SECTIONS_DIR = REPORT_DIR / "secoes"
OUTPUT_PDF = REPORT_DIR / "anderson_correa_deep-learning-and-vision_computer-vision.pdf"

CSS = """
@page { size: a4 portrait; margin: 1.7cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 10pt;
       line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 19pt; color: #0a2540; margin-top: 0.2cm; }
h2 { font-size: 14pt; color: #0a2540; border-bottom: 1px solid #c8d2e0;
     padding-bottom: 3px; margin-top: 0.7cm; }
h3 { font-size: 11.5pt; color: #14457a; margin-top: 0.5cm; }
p, li { text-align: left; }
code { font-family: Courier, monospace; background: #f2f4f7; font-size: 9pt; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 8.5pt; }
th, td { border: 1px solid #b8c2d0; padding: 3px 6px; }
th { background: #e8edf4; }
img { margin: 6px 0; }
"""


def resolve_link(uri: str, _relative: str) -> str:
    """Resolves image paths written relative to the report folder."""
    if uri.startswith(("http://", "https://", "data:")):
        return uri
    return str((REPORT_DIR / uri).resolve())


def merge_sections(sections_dir: Path) -> str:
    """Joins every Markdown section in filename order."""
    parts = [
        path.read_text(encoding="utf-8").strip()
        for path in sorted(Path(sections_dir).glob("*.md"))
    ]
    return "\n\n---\n\n".join(parts) + "\n"


def build(sections_dir: Path, output_pdf: Path) -> None:
    """Renders the merged sections into one PDF file."""
    body = markdown(
        merge_sections(sections_dir),
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    body = re.sub(r"<img ", '<img width="420" ', body)
    html = (
        "<html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )
    with open(output_pdf, "wb") as handle:
        result = pisa.CreatePDF(html, dest=handle, link_callback=resolve_link, encoding="utf-8")
    if result.err:
        raise RuntimeError(f"PDF generation reported {result.err} error(s)")


if __name__ == "__main__":
    build(SECTIONS_DIR, OUTPUT_PDF)
    print(f"wrote {OUTPUT_PDF.relative_to(ROOT)}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_build_report.py -q`
Expected: `2 passed`

- [ ] **Step 5: Write `report/secoes/00_identificacao.md`**

```markdown
# Projeto da Disciplina: Visão Computacional

- **Aluno:** Anderson Corrêa
- **Curso:** Deep Learning and Vision
- **Disciplina:** Computer Vision (INFNET 26E3_3)

## Como ler este relatório

- Cada atividade tem uma seção com: problema, decisões técnicas, resultados e análise crítica.
- Os números e gráficos vêm da execução dos notebooks no Google Colab com GPU T4.
- Cada notebook roda do início ao fim sem alterações e baixa os próprios dados.
```

- [ ] **Step 6: Write `report/secoes/30_atividade3.md`**

Read `results/a3.json` and look at the three figures in `report/figures/` and at ten or more sample images per class (download the dataset locally once with `uv run python -c "from dlv.data import download_kaggle_dataset; download_kaggle_dataset('pavansanagapati/images-dataset', 'data/images_dataset')"`). Then write the section in Portuguese with exactly these headings and contents. Every number must come from `results/a3.json`. Use Brazilian number format (vírgula decimal).

```markdown
# Atividade 3: Classificador com CNN pré-treinada

## Problema

## Decisões técnicas

### Por que ResNet-50

### Por que feature extraction

### Hiperparâmetros

## Resultados

![Curvas de perda e acurácia por época](figures/a3_history.png)

![Acurácia por classe no teste](figures/a3_per_class.png)

![Matriz de confusão no teste](figures/a3_confusion.png)

## Análise crítica

## Melhorias propostas

### Geometric augmentation

### Color augmentation

### Scale variation

### Normalização

## Feature extraction ou fine-tuning
```

Required content per heading. Each bullet below is a check the reviewer applies.

- **Problema:** what is classified, number of images, list of classes with counts, split sizes.
- **Por que ResNet-50** (rubric item 5): fit in a T4 (use `peak_gpu_memory_mb` and `runtime_seconds`), 2,048 features against the number of classes, pre-trained on everyday objects similar to the dataset classes, comparison in one or two bullets with EfficientNet-B0.
- **Por que feature extraction:** small dataset, same domain as ImageNet, risk of overfitting if all 25 million weights train on about 1,260 images. State `trainable_parameters` against `total_parameters`.
- **Hiperparâmetros:** one bullet each for optimizer, learning rate, epochs, batch size, backbone kept in evaluation mode, with the reason for each.
- **Resultados** (rubric item 2): a table with accuracy per class and the global accuracy, plus the three figures. Describe the curves: where validation stabilizes, gap between training and validation.
- **Análise crítica:** which classes are confused with which, read from the confusion matrix, with a likely visual reason for each pair. What the result does not show (single run, single seed, small test set: state how many test images each class has and what one error costs in accuracy).
- **Melhorias propostas** (rubric item 3): for each of the four strategies, state for which classes it helps and for which classes it distorts or harms, using the real class names and what the images look like. At least three strategies must have a class-specific argument. Examples of the kind of reasoning expected: horizontal flip is safe for symmetric objects and harmful where orientation carries meaning; grayscale removes the cue that separates classes distinguished mainly by colour; aggressive random crops can cut out small objects.
- **Feature extraction ou fine-tuning** (rubric item 4): a small decision table with dataset size (small, large) against domain distance from ImageNet (near, far), the recommended strategy in each of the four cells, and where this dataset falls.

Style rules: short sentences, bullet points, plain words. No "não é A, é B" construction. No emojis.

- [ ] **Step 7: Write `notebooks_src/a3/13_analysis.md` and move the archive cell**

```bash
git mv notebooks_src/a3/12_archive.py notebooks_src/a3/14_archive.py
```

`13_analysis.md` holds a short reading of the results in Portuguese, five to eight bullets, taken from the report section: global accuracy, best and worst class, main confusion, what the curves show, and one line pointing to the report for the augmentation discussion. Use the numbers from `results/a3.json`.

- [ ] **Step 8: Update `notebooks_src/a3/00_header.md` with measured values**

Replace the three estimate lines using `results/a3.json`:

- `Memória de GPU`: `peak_gpu_memory_mb` rounded up to the next 0.5 GB, written as "cerca de X GB (medido na T4)"
- `Tempo estimado`: `runtime_seconds` converted to minutes, rounded up, plus 2 minutes for machine start, written as "cerca de N minutos (medido na T4)"
- Keep the `Memória RAM` and `Disco` lines.

- [ ] **Step 9: Run the affected tests**

Run: `uv run pytest tests/test_a3_notebook.py tests/test_build_report.py -q`
Expected: `4 passed`

- [ ] **Step 10: Final Colab run and report build**

Run: `uptime && make test && make colab-a3 && make report && colab sessions`
Expected:
- zero test failures
- Colab run exits 0
- `wrote report/anderson_correa_deep-learning-and-vision_computer-vision.pdf`
- `No active sessions found on server`

The numbers in `results/a3.json` may change slightly from the first run. Reread the file and correct any number in `30_atividade3.md` and `13_analysis.md` that no longer matches, then run `make report` again. If a number in `13_analysis.md` changed, run `make colab-a3` once more so the delivered notebook matches.

- [ ] **Step 11: Check the PDF and the archive**

Run: `make zip && unzip -l entrega/anderson_correa_computer-vision_pd.zip`
Expected: the archive lists `A3_cnn_kaggle.ipynb` and `anderson_correa_deep-learning-and-vision_computer-vision.pdf`.

Open the PDF and confirm: accents render, the three figures appear, the tables have borders, no heading is empty.

- [ ] **Step 12: Commit**

```bash
git add scripts/build_report.py tests/test_build_report.py report/secoes notebooks_src/a3 notebooks/A3_cnn_kaggle.ipynb results/a3.json report/figures report/anderson_correa_deep-learning-and-vision_computer-vision.pdf
git commit -m "Add report builder and Activity 3 report section"
```

---

## What this plan leaves for the next plans

- Activity 1 (paintings, ViT): `attention.py`, `vit.py`, WikiArt preparation, notebook `a1`.
- Activity 4.1 (X-rays, GAN): `gan.py`, notebook `a4`.
- Activity 2 (ads, CLIP): `clip_search.py`, notebook `a2`.
- Final report: remaining sections, Excalidraw diagrams, AI citation section, one browser run of each notebook on Colab, final zip.

Each later plan adds one entry to `NOTEBOOKS` in `scripts/build_notebooks.py`, one `colab-<activity>` target in the `Makefile`, and one cell folder under `notebooks_src/`.
