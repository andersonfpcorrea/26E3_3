"""Tests for run setup and saving."""

import json
import random
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from dlv import run as run_module
from dlv.run import Run, peak_ram_mb, save_run, seed_everything, start_run


def test_seed_everything_makes_random_numbers_repeatable():
    """The same seed gives the same draws from torch, numpy and random."""
    seed_everything(7)
    first = (torch.rand(3), np.random.rand(3), random.random())
    seed_everything(7)
    second = (torch.rand(3), np.random.rand(3), random.random())
    assert torch.equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]


def test_start_run_creates_the_output_folders(tmp_path, monkeypatch):
    """A run starts with its figure folder in place under the current directory."""
    monkeypatch.chdir(tmp_path)
    run = start_run("a9")
    assert isinstance(run, Run)
    assert run.activity == "a9"
    assert run.output_dir == Path("outputs/a9")
    assert run.figures_dir == Path("outputs/a9/figures")
    assert (tmp_path / "outputs" / "a9" / "figures").is_dir()


def test_start_run_can_be_repeated(tmp_path, monkeypatch):
    """Starting the same activity twice reuses the existing folders."""
    monkeypatch.chdir(tmp_path)
    start_run("a9")
    assert start_run("a9").figures_dir.is_dir()


def test_save_run_writes_metrics_with_runtime_and_memory(tmp_path, monkeypatch):
    """The metrics file holds the caller's keys plus runtime and peak memory."""
    monkeypatch.chdir(tmp_path)
    run = start_run("a9")
    save_run(run, {"accuracy": 0.5, "nota": "acurácia"})
    text = (tmp_path / "outputs" / "a9" / "metrics.json").read_text(encoding="utf-8")
    saved = json.loads(text)
    assert saved["accuracy"] == 0.5
    assert saved["nota"] == "acurácia"
    assert saved["runtime_seconds"] >= 0
    assert saved["peak_gpu_memory_mb"] == 0.0
    assert saved["peak_ram_mb"] > 0
    assert set(saved) == {"accuracy", "nota", "runtime_seconds", "peak_gpu_memory_mb", "peak_ram_mb"}
    assert "acurácia" in text
    assert text.startswith('{\n  "accuracy": 0.5')


def test_save_run_measures_time_since_the_run_started(tmp_path, monkeypatch):
    """The saved runtime is the clock difference since the run started."""
    monkeypatch.chdir(tmp_path)
    moments = iter([100.0, 112.34])
    monkeypatch.setattr(run_module, "monotonic", lambda: next(moments))
    run = start_run("a9")
    save_run(run, {})
    saved = json.loads((tmp_path / "outputs" / "a9" / "metrics.json").read_text(encoding="utf-8"))
    assert saved["runtime_seconds"] == 12.3


def test_save_run_does_not_change_the_callers_metrics(tmp_path, monkeypatch):
    """The dict handed in keeps exactly the keys it had."""
    monkeypatch.chdir(tmp_path)
    metrics = {"accuracy": 0.5}
    save_run(start_run("a9"), metrics)
    assert metrics == {"accuracy": 0.5}


def test_save_run_builds_the_archive_layout(tmp_path, monkeypatch):
    """The archive has metrics.json at the top and PNG files under figures/."""
    monkeypatch.chdir(tmp_path)
    run = start_run("a9")
    (run.figures_dir / "history.png").write_bytes(b"png-bytes")
    (run.figures_dir / "notes.txt").write_text("not a figure")
    archive = save_run(run, {"accuracy": 0.5})
    assert archive == Path("outputs_a9.zip")
    with ZipFile(tmp_path / "outputs_a9.zip") as bundle:
        assert sorted(bundle.namelist()) == ["figures/history.png", "metrics.json"]
        assert bundle.read("figures/history.png") == b"png-bytes"
        assert json.loads(bundle.read("metrics.json"))["accuracy"] == 0.5


@pytest.mark.parametrize("platform, expected", [("linux", 2048.0), ("darwin", 2.0)])
def test_peak_ram_reads_the_unit_of_each_platform(monkeypatch, platform, expected):
    """Kilobytes on Linux and bytes on macOS both come out as megabytes."""

    class Usage:
        """Stands in for the resource usage record of a process."""

        ru_maxrss = 2_000_000

    monkeypatch.setattr(run_module.sys, "platform", platform)
    monkeypatch.setattr(run_module, "getrusage", lambda who: Usage())
    assert peak_ram_mb() == expected
