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


def test_strip_rejects_import_with_trailing_comment_paren():
    with pytest.raises(ValueError, match="single-line"):
        strip_package_imports("from dlv.x import a  # (note)\n")


def test_strip_rejects_indented_import():
    with pytest.raises(ValueError, match="single-line"):
        strip_package_imports("def foo():\n    from dlv.x import a\n")


def test_strip_rejects_import_without_trailing_newline():
    with pytest.raises(ValueError, match="single-line"):
        strip_package_imports("from dlv.x import a")


def test_strip_rejects_backslash_continued_import():
    with pytest.raises(ValueError, match="single-line"):
        strip_package_imports("from dlv.x import \\\n    a\n")


def test_strip_allows_dlv_in_docstring_middle():
    source = 'x = """\nThis mentions dlv in the middle\n"""\n'
    assert strip_package_imports(source) == source


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
