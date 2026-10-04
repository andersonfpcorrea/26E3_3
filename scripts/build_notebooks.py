"""Builds self-contained notebooks from cell folders and package modules.

A cell folder holds ordered files. Markdown files become text cells, Python
files become code cells, and ``.modules`` files pull whole modules of the
``dlv`` package into the notebook, one code cell per module, in the order
written. Imports from ``dlv`` are removed, because inside the notebook every
name lives in the same namespace.

Sharing one namespace is also what can break a notebook, so the build checks
it: every removed import must name something that an earlier module cell
provides, and no name may be bound to two different things.
"""

import sys
from argparse import ArgumentParser
from ast import (
    AnnAssign,
    Assign,
    AsyncFunctionDef,
    AugAssign,
    ClassDef,
    FunctionDef,
    Import,
    ImportFrom,
    Name,
    NamedExpr,
    Store,
    parse,
    walk,
)
from dataclasses import dataclass
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = ROOT / "src" / "dlv"
CELLS_DIR = ROOT / "notebooks_src"
BUILD_DIR = ROOT / "build"
EXECUTED_DIR = ROOT / "notebooks"
NOTEBOOKS = {"a1": "A1_vision_transformers", "a2": "A2_clip_ads16", "a3": "A3_cnn_kaggle"}
PACKAGE = "dlv"


@dataclass(frozen=True)
class Source:
    """One cell or module after its package imports were removed.

    ``package_imports`` holds a (module, name) pair for every name a removed
    import expected. ``imports`` maps each name bound by another top-level
    import to where it comes from. ``definitions`` holds the names of the
    top-level functions, classes and assignments.
    """

    filename: str
    text: str
    package_imports: tuple[tuple[str, str], ...]
    imports: dict[str, str]
    definitions: frozenset[str]


def quoted(names: list[str]) -> str:
    """Formats names for an error message, each one in single quotes."""
    return ", ".join(f"'{name}'" for name in names)


def is_package(module: str | None) -> bool:
    """Tells whether a dotted module path is the package or lives inside it."""
    return module == PACKAGE or (module or "").startswith(f"{PACKAGE}.")


def check_plain_import(node: Import, filename: str) -> None:
    """Rejects ``import dlv`` and ``import dlv.<module>``, with or without alias."""
    names = [alias.name for alias in node.names if is_package(alias.name)]
    if names:
        raise ValueError(
            f"{filename}: {quoted(names)} is imported with 'import'; "
            f"use 'from {PACKAGE}.<module> import <names>'"
        )


def check_package_import(node: ImportFrom, filename: str, at_top_level: bool, lines: list[str]) -> None:
    """Rejects every ``from`` import of the package that cannot be removed safely."""
    names = quoted([alias.name for alias in node.names])
    if node.level > 0:
        raise ValueError(
            f"{filename}: {names} comes from a relative import; "
            f"use 'from {PACKAGE}.<module> import <names>'"
        )
    if node.module == PACKAGE:
        raise ValueError(
            f"{filename}: {names} is imported from the package itself; "
            f"use 'from {PACKAGE}.<module> import <names>'"
        )
    if any(alias.asname for alias in node.names):
        raise ValueError(f"{filename}: {names} is imported from {node.module} under an alias")
    if any(alias.name == "*" for alias in node.names):
        raise ValueError(f"{filename}: {names} is imported from {node.module}; name what is needed")
    if not at_top_level:
        raise ValueError(
            f"{filename}: {names} is imported from {node.module} inside a function, class or block; "
            "move the import to the top of the file"
        )
    after = lines[node.end_lineno - 1].encode("utf-8")[node.end_col_offset :].decode("utf-8").strip()
    if node.col_offset > 0 or (after and not after.startswith("#")):
        raise ValueError(
            f"{filename}: the import of {names} from {node.module} shares its line with other code"
        )


def assigned_names(node) -> set[str]:
    """Returns the names a top-level statement binds by assignment.

    Covers plain, annotated and augmented assignments, tuple targets and
    walrus expressions. Loop variables of comprehensions are not bound at
    module level and ``__all__`` only declares exports, so neither counts.
    """
    if isinstance(node, Assign):
        targets = node.targets
    elif isinstance(node, (AnnAssign, AugAssign)):
        targets = [node.target]
    else:
        targets = []
    names = {
        leaf.id
        for target in targets
        for leaf in walk(target)
        if isinstance(leaf, Name) and isinstance(leaf.ctx, Store)
    }
    names.update(leaf.target.id for leaf in walk(node) if isinstance(leaf, NamedExpr))
    names.discard("__all__")
    return names


def bound_names(body: list) -> tuple[dict[str, str], set[str]]:
    """Collects what the top-level statements of a file bind.

    Returns the imported names with their sources, then the names defined by
    functions, classes and assignments. Imports of the package are left out.
    """
    imports = {}
    definitions = set()
    for node in body:
        if isinstance(node, Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                imports[alias.asname or root] = alias.name if alias.asname else root
        elif isinstance(node, ImportFrom) and not is_package(node.module):
            for alias in node.names:
                imports[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, (FunctionDef, AsyncFunctionDef, ClassDef)):
            definitions.add(node.name)
        else:
            definitions.update(assigned_names(node))
    return imports, definitions


def parse_source(source: str, filename: str = "<source>") -> Source:
    """Removes the package imports of a file and records the names it binds.

    Only ``from dlv.<module> import <names>`` at the top level is accepted,
    on one line or several. The statement is cut out by its line range, so the
    rest of the text stays exactly as written. Raises ValueError naming the
    file and the offending name for any other way of importing the package.
    """
    try:
        tree = parse(source)
    except SyntaxError as error:
        raise ValueError(f"{filename}: cannot be parsed as Python: {error}") from error
    lines = source.split("\n")
    top_level = {id(node) for node in tree.body}
    package_imports = []
    removed = set()
    for node in walk(tree):
        if isinstance(node, Import):
            check_plain_import(node, filename)
        elif isinstance(node, ImportFrom) and (node.level > 0 or is_package(node.module)):
            check_package_import(node, filename, id(node) in top_level, lines)
            module = node.module.removeprefix(f"{PACKAGE}.")
            package_imports.extend((module, alias.name) for alias in node.names)
            removed.update(range(node.lineno - 1, node.end_lineno))
    kept = "\n".join(line for number, line in enumerate(lines) if number not in removed)
    imports, definitions = bound_names(tree.body)
    return Source(
        filename=filename,
        text=kept.strip() + "\n",
        package_imports=tuple(package_imports),
        imports=imports,
        definitions=frozenset(definitions),
    )


def strip_package_imports(source: str, filename: str = "<source>") -> str:
    """Returns source without its ``from dlv.<module> import <names>`` statements.

    Raises ValueError for import forms that cannot be removed safely.
    """
    return parse_source(source, filename).text


class Namespace:
    """The names bound so far by the cells of one notebook, in cell order.

    Every cell of a notebook shares one namespace, so a name bound twice to
    different things silently replaces the first one. Sources are added in
    cell order and the first conflict raises ValueError.
    """

    def __init__(self) -> None:
        """Starts with no module inlined and no name bound."""
        self._modules = {}
        self._origins = {}

    def add(self, source: Source, module: str | None = None) -> None:
        """Checks one source against the earlier cells and records its names.

        ``module`` is the package module the source inlines, or None for a
        cell file. Top-level definitions are tracked for modules only, since
        cell files are free to reassign their own variables.
        """
        self._resolve(source)
        for name, origin in source.imports.items():
            self._claim(name, ("import", origin), source.filename)
        if module is None:
            return
        for name in sorted(source.definitions):
            self._claim(name, ("definition", source.filename), source.filename)
        provided = {name for _, name in source.package_imports}
        self._modules[module] = provided | set(source.imports) | source.definitions

    def _resolve(self, source: Source) -> None:
        """Checks that every removed import names something an earlier module provides."""
        for module, name in source.package_imports:
            if module not in self._modules:
                raise ValueError(
                    f"{source.filename}: '{name}' is imported from {PACKAGE}.{module}, "
                    "but no earlier cell inlines that module"
                )
            if name not in self._modules[module]:
                raise ValueError(
                    f"{source.filename}: '{name}' is imported from {PACKAGE}.{module}, "
                    "which does not define it at top level"
                )

    def _claim(self, name: str, origin: tuple[str, str], filename: str) -> None:
        """Binds a name to its origin, or raises if another file bound it differently."""
        known_origin, known_filename = self._origins.setdefault(name, (origin, filename))
        if known_origin != origin and known_filename != filename:
            raise ValueError(
                f"{filename}: '{name}' is {describe(origin)}, "
                f"but {known_filename} already made it {describe(known_origin)}"
            )


def describe(origin: tuple[str, str]) -> str:
    """Words the origin of a name for an error message."""
    kind, detail = origin
    return f"an import of {detail}" if kind == "import" else f"a definition in {detail}"


def code_cell(path: Path, namespace: Namespace, module: str | None = None):
    """Turns one Python file into a checked code cell without package imports."""
    source = parse_source(path.read_text(encoding="utf-8"), path.name)
    namespace.add(source, module)
    return new_code_cell(source.text)


def build_notebook(cells_dir: Path, package_dir: Path) -> nbformat.NotebookNode:
    """Assembles a notebook from the files of a cell folder, in filename order.

    Raises ValueError when a package import cannot be removed, when it names
    something no earlier module cell provides, or when two files bind the same
    name to different things.
    """
    namespace = Namespace()
    cells = []
    for path in sorted(Path(cells_dir).iterdir()):
        if path.suffix == ".md":
            cells.append(new_markdown_cell(path.read_text(encoding="utf-8").strip()))
        elif path.suffix == ".py":
            cells.append(code_cell(path, namespace))
        elif path.suffix == ".modules":
            for name in path.read_text(encoding="utf-8").split():
                cells.append(code_cell(Path(package_dir) / f"{name}.py", namespace, module=name))
    metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
    }
    return new_notebook(cells=cells, metadata=metadata)


def cell_sources(notebook: nbformat.NotebookNode) -> list[tuple[str, str]]:
    """Returns the type and source of every cell, which is what a build decides."""
    return [(cell.cell_type, cell.source) for cell in notebook.cells]


def stale_notebooks(
    cells_dir: Path | None = None,
    package_dir: Path | None = None,
    executed_dir: Path | None = None,
    notebooks: dict[str, str] | None = None,
) -> list[str]:
    """Returns the names of notebooks whose executed copy no longer matches a fresh build.

    Cell types and sources are compared, outputs are ignored. A notebook with
    no executed copy is stale. The arguments default to the project folders.
    """
    cells_dir = CELLS_DIR if cells_dir is None else Path(cells_dir)
    package_dir = PACKAGE_DIR if package_dir is None else Path(package_dir)
    executed_dir = EXECUTED_DIR if executed_dir is None else Path(executed_dir)
    stale = []
    for activity, name in (NOTEBOOKS if notebooks is None else notebooks).items():
        executed = executed_dir / f"{name}.ipynb"
        fresh = cell_sources(build_notebook(cells_dir / activity, package_dir))
        if not executed.exists() or cell_sources(nbformat.read(executed, as_version=4)) != fresh:
            stale.append(name)
    return stale


def write_notebooks() -> None:
    """Writes every notebook listed in NOTEBOOKS into ``build/notebooks``."""
    output_dir = BUILD_DIR / "notebooks"
    output_dir.mkdir(parents=True, exist_ok=True)
    for activity, name in NOTEBOOKS.items():
        notebook = build_notebook(CELLS_DIR / activity, PACKAGE_DIR)
        target = output_dir / f"{name}.ipynb"
        nbformat.write(notebook, target)
        print(f"wrote {target.relative_to(ROOT)}")


def main(arguments: list[str] | None = None) -> int:
    """Builds the notebooks, or with ``--check`` reports the stale ones.

    Returns the process exit code: 1 when ``--check`` finds a stale notebook,
    0 otherwise.
    """
    parser = ArgumentParser(description="Build the notebooks into build/notebooks.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; list executed notebooks that differ from a fresh build",
    )
    if not parser.parse_args(arguments).check:
        write_notebooks()
        return 0
    stale = stale_notebooks()
    for name in stale:
        print(f"stale: {name}")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
