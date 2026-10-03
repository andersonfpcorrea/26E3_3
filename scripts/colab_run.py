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
) -> bool:
    """Copies metrics and figures from a run archive into the repository.

    Returns True if metrics.json was found in the archive, False otherwise.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_found = False
    with ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if name == "metrics.json":
                (results_dir / f"{activity}.json").write_bytes(bundle.read(name))
                metrics_found = True
            elif name.startswith("figures/") and name.endswith(".png"):
                target = figures_dir / f"{activity}_{Path(name).name}"
                target.write_bytes(bundle.read(name))
    return metrics_found


def _stop_session_safely(session: str, execute) -> None:
    """Stops a Colab session, catching and logging errors without raising."""
    try:
        execute(["colab", "stop", "-s", session], check=False)
    except OSError as error:
        print(f"warning: could not stop session {session}: {error}", file=sys.stderr)


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
            ["colab", "exec", "-s", session, "-f", f"{name}.ipynb", "--timeout", "1800"],
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
        _stop_session_safely(session, execute)
    metrics_found = collect_outputs(archive, activity, ROOT / "results", ROOT / "report" / "figures")
    if not metrics_found:
        print(f"colab run failed: no metrics.json in archive for {activity}", file=sys.stderr)
        return 1
    (ROOT / "notebooks").mkdir(exist_ok=True)
    copyfile(executed, ROOT / "notebooks" / f"{name}.ipynb")
    return 0


if __name__ == "__main__":
    sys.exit(run_on_colab(sys.argv[1]))
