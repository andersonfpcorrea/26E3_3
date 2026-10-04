"""Tests for the notebook builder."""

import nbformat
import pytest
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

import build_notebooks
from build_notebooks import build_notebook, main, stale_notebooks, strip_package_imports

BETA = '"""Beta module."""\n\n\ndef thing():\n    """Returns two."""\n    return 2\n'
ALPHA = (
    '"""Alpha module."""\n\nfrom dlv.beta import thing\n\n\n'
    'def alpha():\n    """Returns one."""\n    return thing() - 1\n'
)


def make_sources(tmp_path):
    """Creates a tiny package folder and a tiny cell folder for the builder."""
    package = tmp_path / "pkg"
    cells = tmp_path / "cells"
    package.mkdir()
    cells.mkdir()
    (package / "beta.py").write_text(BETA)
    (package / "alpha.py").write_text(ALPHA)
    (cells / "00_intro.md").write_text("# Titulo\n")
    (cells / "01_support.modules").write_text("beta\nalpha\n")
    (cells / "02_run.py").write_text("from dlv.alpha import alpha\n\nprint(alpha())\n")
    (cells / ".DS_Store").write_text("ignored")
    return cells, package


def test_strip_removes_single_line_package_imports():
    """A one-line package import disappears and the other lines stay."""
    source = "from pathlib import Path\nfrom dlv.data import load\n\nx = 1\n"
    assert strip_package_imports(source) == "from pathlib import Path\n\nx = 1\n"


def test_strip_removes_multiline_package_import_and_keeps_surrounding_text():
    """A parenthesised import is removed and every other character survives."""
    source = (
        '"""Docstring  with   odd spacing."""\n'
        "\n"
        "from pathlib import Path\n"
        "from dlv.data import (\n"
        "    load,\n"
        "    save,\n"
        ")\n"
        "\n"
        "\n"
        "def run( a,b ):\n"
        '    """Keeps   this text."""\n'
        "    return {'a':a,\n"
        "            'b':b}\n"
    )
    expected = (
        '"""Docstring  with   odd spacing."""\n'
        "\n"
        "from pathlib import Path\n"
        "\n"
        "\n"
        "def run( a,b ):\n"
        '    """Keeps   this text."""\n'
        "    return {'a':a,\n"
        "            'b':b}\n"
    )
    assert strip_package_imports(source) == expected


def test_strip_removes_import_with_trailing_comment():
    """A package import followed by a comment is removed with its comment."""
    assert strip_package_imports("x = 1\nfrom dlv.x import a  # (note)\ny = 2\n") == "x = 1\ny = 2\n"


def test_strip_removes_import_without_trailing_newline():
    """A package import on an unterminated last line is removed."""
    assert strip_package_imports("x = 1\nfrom dlv.x import a") == "x = 1\n"


def test_strip_removes_backslash_continued_import():
    """A package import continued with a backslash is removed whole."""
    assert strip_package_imports("from dlv.x import \\\n    a\nx = 1\n") == "x = 1\n"


def test_strip_allows_dlv_in_docstring_middle():
    """Text that only mentions the package inside a string is left alone."""
    source = 'x = """\nThis mentions dlv in the middle\nfrom dlv.data import load\n"""\n'
    assert strip_package_imports(source) == source


@pytest.mark.parametrize(
    "source, name",
    [
        ("from .x import y\n", "y"),
        ("from . import x\n", "x"),
        ("from dlv import data\n", "data"),
        ("import dlv\n", "dlv"),
        ("import dlv.data\n", "dlv.data"),
        ("import dlv.data as data\n", "dlv.data"),
        ("from dlv.data import load as fetch\n", "load"),
        ("from dlv.data import *\n", "*"),
        ("def foo():\n    from dlv.x import a\n", "a"),
        ("class Foo:\n    from dlv.x import a\n", "a"),
        ("if True:\n    from dlv.x import a\n", "a"),
        ("try:\n    from dlv.x import a\nexcept ImportError:\n    a = None\n", "a"),
        ("x = 1; from dlv.x import a\n", "a"),
        ("from dlv.x import a; x = 1\n", "a"),
    ],
)
def test_strip_rejects_unsupported_package_import(source, name):
    """Each unsupported import form fails, naming the file and the name."""
    with pytest.raises(ValueError) as failure:
        strip_package_imports(source, "cell.py")
    assert "cell.py" in str(failure.value)
    assert f"'{name}'" in str(failure.value)


def test_strip_names_the_file_with_a_syntax_error():
    """Source that does not parse fails with the file name in the message."""
    with pytest.raises(ValueError, match="broken.py"):
        strip_package_imports("def broken(:\n", "broken.py")


def test_build_orders_cells_by_filename_and_inlines_modules(tmp_path):
    """Cells follow filename order and listed modules become code cells."""
    cells, package = make_sources(tmp_path)
    notebook = build_notebook(cells, package)
    assert [cell.cell_type for cell in notebook.cells] == ["markdown", "code", "code", "code"]
    assert notebook.cells[0].source == "# Titulo"
    assert "def thing" in notebook.cells[1].source
    assert "def alpha" in notebook.cells[2].source
    assert "print(alpha())" in notebook.cells[3].source


def test_built_notebook_has_no_package_imports(tmp_path):
    """No package import survives in the built notebook."""
    cells, package = make_sources(tmp_path)
    notebook = build_notebook(cells, package)
    assert "from dlv" not in "".join(cell.source for cell in notebook.cells)


def test_built_notebook_runs_on_its_own(tmp_path):
    """The joined code of a built notebook executes without the package."""
    cells, package = make_sources(tmp_path)
    notebook = build_notebook(cells, package)
    namespace = {}
    exec("\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code"), namespace)
    assert namespace["alpha"]() == 1


def test_built_notebook_is_valid_and_requests_t4(tmp_path):
    """The notebook passes schema validation and asks Colab for a T4."""
    cells, package = make_sources(tmp_path)
    notebook = build_notebook(cells, package)
    nbformat.validate(notebook)
    assert notebook.metadata["accelerator"] == "GPU"
    assert notebook.metadata["colab"]["gpuType"] == "T4"


def test_build_fails_when_module_is_missing(tmp_path):
    """A listing that names a module without a file fails."""
    cells, package = make_sources(tmp_path)
    (cells / "01_support.modules").write_text("missing\n")
    with pytest.raises(FileNotFoundError):
        build_notebook(cells, package)


def test_build_fails_when_imported_name_is_not_defined(tmp_path):
    """A cell importing a name its module does not define fails."""
    cells, package = make_sources(tmp_path)
    (cells / "02_run.py").write_text("from dlv.alpha import alpha, gamma\n\nprint(alpha())\n")
    with pytest.raises(ValueError) as failure:
        build_notebook(cells, package)
    assert "02_run.py" in str(failure.value)
    assert "'gamma'" in str(failure.value)


def test_build_fails_when_module_comes_after_the_cell_that_needs_it(tmp_path):
    """A cell that sorts before the listing of its module fails."""
    cells, package = make_sources(tmp_path)
    (cells / "02_run.py").rename(cells / "00_run.py")
    with pytest.raises(ValueError) as failure:
        build_notebook(cells, package)
    assert "00_run.py" in str(failure.value)
    assert "'alpha'" in str(failure.value)


def test_build_fails_when_module_is_listed_before_its_dependency(tmp_path):
    """A module listed ahead of the module it imports from fails."""
    cells, package = make_sources(tmp_path)
    (cells / "01_support.modules").write_text("alpha\nbeta\n")
    with pytest.raises(ValueError) as failure:
        build_notebook(cells, package)
    assert "alpha.py" in str(failure.value)
    assert "'thing'" in str(failure.value)


def test_build_fails_when_two_modules_define_the_same_function(tmp_path):
    """Two inlined modules defining one name fail, naming both files."""
    cells, package = make_sources(tmp_path)
    (package / "gamma.py").write_text('def thing():\n    """Returns three."""\n    return 3\n')
    (cells / "01_support.modules").write_text("beta\nalpha\ngamma\n")
    with pytest.raises(ValueError) as failure:
        build_notebook(cells, package)
    assert "gamma.py" in str(failure.value)
    assert "beta.py" in str(failure.value)
    assert "'thing'" in str(failure.value)


def test_build_fails_when_two_modules_assign_the_same_name(tmp_path):
    """Two inlined modules assigning one constant fail."""
    cells, package = make_sources(tmp_path)
    (package / "beta.py").write_text(BETA + "\nLIMIT = 1\n")
    (package / "gamma.py").write_text("LIMIT, OTHER = 2, 3\n")
    (cells / "01_support.modules").write_text("beta\nalpha\ngamma\n")
    with pytest.raises(ValueError, match="'LIMIT'"):
        build_notebook(cells, package)


def test_build_fails_when_a_name_is_imported_from_two_sources(tmp_path):
    """One name imported from different sources in a module and a cell fails."""
    cells, package = make_sources(tmp_path)
    (package / "beta.py").write_text("from PIL import Image\n" + BETA)
    (cells / "02_run.py").write_text(
        "from IPython.display import Image\nfrom dlv.alpha import alpha\n\nprint(alpha())\n"
    )
    with pytest.raises(ValueError) as failure:
        build_notebook(cells, package)
    assert "02_run.py" in str(failure.value)
    assert "'Image'" in str(failure.value)
    assert "PIL.Image" in str(failure.value)
    assert "IPython.display.Image" in str(failure.value)


def test_build_fails_when_a_module_defines_a_name_another_imports(tmp_path):
    """A module defining a name that another module imports fails."""
    cells, package = make_sources(tmp_path)
    (package / "gamma.py").write_text('def Path():\n    """Returns nothing."""\n')
    (package / "beta.py").write_text("from pathlib import Path\n" + BETA)
    (cells / "01_support.modules").write_text("beta\nalpha\ngamma\n")
    with pytest.raises(ValueError, match="'Path'"):
        build_notebook(cells, package)


def test_build_accepts_one_name_imported_from_the_same_source_twice(tmp_path):
    """The same import repeated in modules and cells is fine."""
    cells, package = make_sources(tmp_path)
    (package / "beta.py").write_text("import numpy as np\nfrom pathlib import Path\n" + BETA)
    (cells / "02_run.py").write_text(
        "import numpy as np\nfrom pathlib import Path\nfrom dlv.alpha import alpha\n\nprint(alpha())\n"
    )
    notebook = build_notebook(cells, package)
    assert notebook.cells[3].source == "import numpy as np\nfrom pathlib import Path\n\nprint(alpha())\n"


def test_cell_may_import_a_name_its_module_imported(tmp_path):
    """A name a module imported at top level can be imported from that module."""
    cells, package = make_sources(tmp_path)
    (cells / "02_run.py").write_text("from dlv.alpha import alpha, thing\n\nprint(alpha(), thing())\n")
    notebook = build_notebook(cells, package)
    assert notebook.cells[3].source == "print(alpha(), thing())\n"


def write_executed(tmp_path, cells, package, name="Demo"):
    """Writes the built notebook where executed notebooks are kept."""
    executed = tmp_path / "notebooks"
    executed.mkdir(exist_ok=True)
    nbformat.write(build_notebook(cells, package), executed / f"{name}.ipynb")
    return executed


def test_stale_notebooks_is_empty_when_executed_copy_matches(tmp_path):
    """An executed copy with the same cells as a fresh build is not stale."""
    cells, package = make_sources(tmp_path)
    executed = write_executed(tmp_path, cells, package)
    assert stale_notebooks(tmp_path, package, executed, {"cells": "Demo"}) == []


def test_stale_notebooks_ignores_outputs(tmp_path):
    """Outputs and execution counts do not make a notebook stale."""
    cells, package = make_sources(tmp_path)
    executed = write_executed(tmp_path, cells, package)
    notebook = nbformat.read(executed / "Demo.ipynb", as_version=4)
    notebook.cells[1].execution_count = 7
    nbformat.write(notebook, executed / "Demo.ipynb")
    assert stale_notebooks(tmp_path, package, executed, {"cells": "Demo"}) == []


def test_stale_notebooks_reports_changed_source(tmp_path):
    """A cell file edited after the run makes the notebook stale."""
    cells, package = make_sources(tmp_path)
    executed = write_executed(tmp_path, cells, package)
    (cells / "00_intro.md").write_text("# Outro titulo\n")
    assert stale_notebooks(tmp_path, package, executed, {"cells": "Demo"}) == ["Demo"]


def test_stale_notebooks_reports_changed_cell_type(tmp_path):
    """A cell whose type changed makes the notebook stale."""
    cells, package = make_sources(tmp_path)
    executed = tmp_path / "notebooks"
    executed.mkdir()
    built = build_notebook(cells, package)
    swapped = [new_code_cell(built.cells[0].source), *built.cells[1:]]
    nbformat.write(new_notebook(cells=swapped), executed / "Demo.ipynb")
    assert stale_notebooks(tmp_path, package, executed, {"cells": "Demo"}) == ["Demo"]


def test_stale_notebooks_reports_extra_cell(tmp_path):
    """An executed copy with one more cell than the build is stale."""
    cells, package = make_sources(tmp_path)
    executed = tmp_path / "notebooks"
    executed.mkdir()
    built = build_notebook(cells, package)
    nbformat.write(
        new_notebook(cells=[*built.cells, new_markdown_cell("extra")]), executed / "Demo.ipynb"
    )
    assert stale_notebooks(tmp_path, package, executed, {"cells": "Demo"}) == ["Demo"]


def test_stale_notebooks_reports_missing_executed_copy(tmp_path):
    """A notebook that was never run counts as stale."""
    cells, package = make_sources(tmp_path)
    assert stale_notebooks(tmp_path, package, tmp_path / "notebooks", {"cells": "Demo"}) == ["Demo"]


def point_builder_at(tmp_path, monkeypatch, package):
    """Makes the builder defaults use temporary folders and one notebook."""
    monkeypatch.setattr(build_notebooks, "ROOT", tmp_path)
    monkeypatch.setattr(build_notebooks, "CELLS_DIR", tmp_path)
    monkeypatch.setattr(build_notebooks, "PACKAGE_DIR", package)
    monkeypatch.setattr(build_notebooks, "BUILD_DIR", tmp_path / "build")
    monkeypatch.setattr(build_notebooks, "EXECUTED_DIR", tmp_path / "notebooks")
    monkeypatch.setattr(build_notebooks, "NOTEBOOKS", {"cells": "Demo"})


def test_main_writes_notebooks_into_the_build_folder(tmp_path, monkeypatch):
    """Without flags the builder writes each notebook and exits with 0."""
    cells, package = make_sources(tmp_path)
    point_builder_at(tmp_path, monkeypatch, package)
    assert main([]) == 0
    assert (tmp_path / "build" / "notebooks" / "Demo.ipynb").exists()


def test_main_check_exits_with_1_and_prints_stale_names(tmp_path, monkeypatch, capsys):
    """The check flag prints stale notebooks, exits with 1 and writes nothing."""
    cells, package = make_sources(tmp_path)
    point_builder_at(tmp_path, monkeypatch, package)
    assert main(["--check"]) == 1
    assert "Demo" in capsys.readouterr().out
    assert not (tmp_path / "build").exists()


def test_main_check_exits_with_0_when_nothing_is_stale(tmp_path, monkeypatch):
    """The check flag exits with 0 when every executed copy is fresh."""
    cells, package = make_sources(tmp_path)
    point_builder_at(tmp_path, monkeypatch, package)
    write_executed(tmp_path, cells, package)
    assert main(["--check"]) == 0


def test_build_accepts_the_same_comprehension_variable_in_two_modules(tmp_path):
    """Comprehension loop variables are not definitions, so sharing one is fine."""
    cells, package = make_sources(tmp_path)
    (package / "beta.py").write_text(BETA + "\nSQUARES = [item * item for item in range(3)]\n")
    (package / "gamma.py").write_text("CUBES = [item**3 for item in range(3)]\n")
    (cells / "01_support.modules").write_text("beta\nalpha\ngamma\n")
    build_notebook(cells, package)


def test_build_accepts_dunder_all_in_two_modules(tmp_path):
    """Each module may declare its own ``__all__``."""
    cells, package = make_sources(tmp_path)
    (package / "beta.py").write_text(BETA + '\n__all__ = ["thing"]\n')
    (package / "gamma.py").write_text('__all__ = ["other"]\n\n\ndef other():\n    """Returns four."""\n    return 4\n')
    (cells / "01_support.modules").write_text("beta\nalpha\ngamma\n")
    build_notebook(cells, package)


@pytest.mark.parametrize(
    "line",
    ["LIMIT: int = 2", "LIMIT += 2", "(LIMIT := 2)", "(LIMIT, OTHER) = 2, 3", "LIMIT = OTHER = 2"],
)
def test_build_fails_when_two_modules_bind_the_same_name_in_any_form(tmp_path, line):
    """Annotated, augmented, walrus, tuple and chained bindings all count."""
    cells, package = make_sources(tmp_path)
    (package / "beta.py").write_text(BETA + "\nLIMIT = 1\n")
    (package / "gamma.py").write_text(line + "\n")
    (cells / "01_support.modules").write_text("beta\nalpha\ngamma\n")
    with pytest.raises(ValueError, match="'LIMIT'"):
        build_notebook(cells, package)
