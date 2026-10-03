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
    r"^(import dlv|from dlv(\.\w+)* import \(|from dlv(\.\w+)* import [^(\n]*\\)", re.MULTILINE
)
REMAINING_IMPORT = re.compile(r"^\s*(from dlv\b|import dlv\b)", re.MULTILINE)


def strip_package_imports(source: str) -> str:
    """Removes single-line ``from dlv... import ...`` statements from source.

    Raises ValueError for import forms that cannot be removed safely.
    """
    if UNSUPPORTED_IMPORT.search(source):
        raise ValueError(
            "only single-line 'from dlv.<module> import <names>' imports are supported"
        )
    result = PACKAGE_IMPORT.sub("", source).strip() + "\n"
    if REMAINING_IMPORT.search(result):
        raise ValueError(
            "only single-line 'from dlv.<module> import <names>' imports are supported"
        )
    return result


def module_cells(listing: str, package_dir: Path) -> list:
    """Returns one code cell per module named in a ``.modules`` listing."""
    names = [line.strip() for line in listing.splitlines() if line.strip()]
    return [
        new_code_cell(strip_package_imports((package_dir / f"{name}.py").read_text(encoding="utf-8")))
        for name in names
    ]


def build_notebook(cells_dir: Path, package_dir: Path) -> nbformat.NotebookNode:
    """Assembles a notebook from the files of a cell folder, in filename order."""
    cells = []
    for path in sorted(Path(cells_dir).iterdir()):
        text = path.read_text(encoding="utf-8") if path.suffix in {".md", ".py", ".modules"} else ""
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
