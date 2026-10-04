"""Run setup and saving shared by the notebooks.

A run seeds the random generators, keeps its figures in one folder and ends
by writing its metrics and packing everything into one archive. The archive
layout is the contract with the script that collects the outputs of a Colab
run: ``metrics.json`` at the top level and the PNG files under ``figures/``.
"""

import json
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from resource import RUSAGE_SELF, getrusage
from time import monotonic
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch

BYTES_PER_MEGABYTE = 1e6


class Stopwatch:
    """Measures how long the named steps of a run take.

    ``seconds`` maps every finished step to its duration, in the order the
    steps ran, ready to be saved with the metrics. Each step is also printed
    when it ends, so the cost of a notebook is visible while it runs.
    """

    def __init__(self) -> None:
        """Starts with no step measured."""
        self.seconds = {}

    @contextmanager
    def measure(self, name: str):
        """Times the body of a ``with`` block and records it under ``name``.

        A block that raises is not recorded.
        """
        started_at = monotonic()
        yield
        self.seconds[name] = round(monotonic() - started_at, 1)
        print(f"{name}: {self.seconds[name]} s")


@dataclass(frozen=True)
class Run:
    """One notebook run: its activity, its output folders and when it started.

    ``started_at`` is a reading of the monotonic clock, good only for
    measuring elapsed time.
    """

    activity: str
    output_dir: Path
    figures_dir: Path
    started_at: float


def seed_everything(seed: int) -> None:
    """Seeds the random generators of Python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def start_run(activity: str) -> Run:
    """Prepares the output folders of an activity and starts its timer.

    Creates ``outputs/<activity>/figures`` under the current directory. The
    peak GPU memory counter is reset when CUDA is available, so the peak saved
    later belongs to this run only.
    """
    output_dir = Path("outputs") / activity
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return Run(activity, output_dir, figures_dir, monotonic())


def peak_gpu_memory_mb() -> float:
    """Returns the peak memory allocated by PyTorch on the GPU, or 0.0 without CUDA."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / BYTES_PER_MEGABYTE


def peak_ram_mb() -> float:
    """Returns the peak resident memory of the main process, in megabytes.

    Data loader worker processes are not included. The operating system
    reports the value in kilobytes on Linux and in bytes on macOS.
    """
    peak = getrusage(RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    return peak_bytes / BYTES_PER_MEGABYTE


def save_run(run: Run, metrics: dict) -> Path:
    """Writes the metrics of a run and packs them with its figures.

    A copy of ``metrics`` gains ``runtime_seconds``, ``peak_gpu_memory_mb``
    and ``peak_ram_mb`` and is written to ``outputs/<activity>/metrics.json``.
    The archive ``outputs_<activity>.zip`` is created in the current directory
    and its path is returned.
    """
    saved = {
        **metrics,
        "runtime_seconds": round(monotonic() - run.started_at, 1),
        "peak_gpu_memory_mb": round(peak_gpu_memory_mb(), 1),
        "peak_ram_mb": round(peak_ram_mb(), 1),
    }
    metrics_path = run.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(saved, indent=2, ensure_ascii=False), encoding="utf-8")
    archive = Path(f"outputs_{run.activity}.zip")
    with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
        bundle.write(metrics_path, "metrics.json")
        for figure in sorted(run.figures_dir.glob("*.png")):
            bundle.write(figure, f"figures/{figure.name}")
    return archive
