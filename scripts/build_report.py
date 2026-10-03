"""Joins the report sections and renders the PDF deliverable.

Sections live in ``report/secoes`` and are joined in filename order, so the
numeric prefixes define the structure of the document. The toolchain is pure
Python: ``markdown`` for HTML and ``xhtml2pdf`` for the PDF.
"""

import re
from pathlib import Path

from markdown import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "report"
SECTIONS_DIR = REPORT_DIR / "secoes"
OUTPUT_PDF = REPORT_DIR / "anderson_correa_deep-learning-and-vision_computer-vision.pdf"

CSS = """
@page { size: a4 portrait; margin: 1.7cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 10pt;
       line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 19pt; color: #0a2540; margin-top: 0.2cm; }
h2 { font-size: 14pt; color: #0a2540; border-bottom: 1px solid #c8d2e0;
     padding-bottom: 3px; margin-top: 0.7cm; }
h3 { font-size: 11.5pt; color: #14457a; margin-top: 0.5cm; }
p, li { text-align: left; }
code { font-family: Courier, monospace; background: #f2f4f7; font-size: 9pt; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 8.5pt; }
th, td { border: 1px solid #b8c2d0; padding: 3px 6px; }
th { background: #e8edf4; }
img { margin: 6px 0; }
"""


def resolve_link(uri: str, _relative: str) -> str:
    """Resolves image paths written relative to the report folder."""
    if uri.startswith(("http://", "https://", "data:")):
        return uri
    return str((REPORT_DIR / uri).resolve())


def merge_sections(sections_dir: Path) -> str:
    """Joins every Markdown section in filename order."""
    parts = [
        path.read_text(encoding="utf-8").strip()
        for path in sorted(Path(sections_dir).glob("*.md"))
    ]
    return "\n\n---\n\n".join(parts) + "\n"


def build(sections_dir: Path, output_pdf: Path) -> None:
    """Renders the merged sections into one PDF file."""
    body = markdown(
        merge_sections(sections_dir),
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    body = re.sub(r"<img ", '<img width="420" ', body)
    html = (
        "<html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )
    with open(output_pdf, "wb") as handle:
        result = pisa.CreatePDF(html, dest=handle, link_callback=resolve_link, encoding="utf-8")
    if result.err:
        raise RuntimeError(f"PDF generation reported {result.err} error(s)")


if __name__ == "__main__":
    build(SECTIONS_DIR, OUTPUT_PDF)
    print(f"wrote {OUTPUT_PDF.relative_to(ROOT)}")
