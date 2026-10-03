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
