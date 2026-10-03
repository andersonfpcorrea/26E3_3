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
