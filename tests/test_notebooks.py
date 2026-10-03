"""Checks that every notebook builds into a self-contained, runnable notebook."""

from ast import Import, ImportFrom, parse, walk

import pytest

from build_notebooks import CELLS_DIR, NOTEBOOKS, PACKAGE_DIR, build_notebook


@pytest.fixture(params=sorted(NOTEBOOKS))
def notebook(request):
    """Builds the notebook of one activity, once for every activity."""
    return build_notebook(CELLS_DIR / request.param, PACKAGE_DIR)


def joined_code(notebook):
    """Returns the source of every code cell as one program."""
    return "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")


def test_notebook_code_compiles(notebook):
    """The code cells, taken together, are valid Python."""
    compile(joined_code(notebook), "notebook", "exec")


def test_notebook_has_no_package_import(notebook):
    """No import statement in the notebook reaches for this repository."""
    imported = []
    for node in walk(parse(joined_code(notebook))):
        if isinstance(node, Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ImportFrom):
            imported.append("." * node.level + (node.module or ""))
    assert imported
    assert not [name for name in imported if name.startswith(".") or name.split(".")[0] == "dlv"]


def test_notebook_starts_with_time_and_memory_needs(notebook):
    """The first cell is text that states the run time and the memory needed."""
    first = notebook.cells[0]
    assert first.cell_type == "markdown"
    assert "Tempo estimado" in first.source
    assert "Memória" in first.source


def test_notebook_ends_by_saving_the_run(notebook):
    """The last cell is code that calls the run-saving helper."""
    last = notebook.cells[-1]
    assert last.cell_type == "code"
    assert "save_run(" in last.source
