"""Joins the report sections and renders the PDF deliverable.

Sections live in ``report/secoes`` and are joined in filename order, so the
numeric prefixes define the structure of the document. The toolchain is pure
Python: ``markdown`` for HTML and ``xhtml2pdf`` for the PDF.

Image sizes keep each file's aspect ratio. The width follows a rule on the
width to height ratio of the file: a strip at least 2.4 times wider than tall
(multi-panel curves, image grids) takes the full text width, and any other
figure (single charts, confusion matrices) takes two thirds of it. A figure
taller than the page limit shrinks until it fits. To override the rule, write
the width in points after the image in the Markdown line, for example
``![caption](figures/a.png){: width="499"}``; the height still follows the
file's ratio. Each image is embedded at about ``PRINT_DPI`` dots per inch at its
displayed width, using a downscaled copy written to the build folder; the files
under ``report`` are never modified and small images are never upscaled.
Widths are computed in points and written to the HTML in pixels (4/3 of the
points), because the PDF library reads a width attribute as pixels. The alt text
of each image is printed as a caption under it, kept on the same page. Every
top-level heading after the first starts a new page, and a paragraph made only
of bold text stays with the block that follows. Headings are kept on the same
page as the block that follows.
"""

import re
from hashlib import sha1
from pathlib import Path

from markdown import markdown
from PIL import Image
from xhtml2pdf import pisa
from xhtml2pdf.config.resources import ResourceAccessPolicy, use_policy

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "report"
SECTIONS_DIR = REPORT_DIR / "secoes"
OUTPUT_PDF = REPORT_DIR / "anderson_correa_deep-learning-and-vision_computer-vision.pdf"

FULL_WIDTH = 499
SINGLE_WIDTH = 333
MAX_HEIGHT = 690
PIXELS_PER_POINT = 4 / 3
STRIP_RATIO = 2.4
PRINT_DPI = 150
PHOTO_COLORS = 4000
JPEG_QUALITY = 85

CSS = """
@page { size: a4 portrait; margin: 1.7cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 10pt;
       line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 19pt; color: #0a2540; margin-top: 0.2cm; -pdf-keep-with-next: true; }
h2 { font-size: 14pt; color: #0a2540; border-bottom: 1px solid #c8d2e0;
     padding-bottom: 3px; margin-top: 0.7cm; -pdf-keep-with-next: true; }
h3 { font-size: 11.5pt; color: #14457a; margin-top: 0.5cm; -pdf-keep-with-next: true; }
p, li { text-align: left; }
code { font-family: Courier, monospace; background: #f2f4f7; font-size: 9pt; }
table { width: 100%; margin: 8px 0; font-size: 8.5pt; }
th, td { border: 1px solid #b8c2d0; padding: 3px 6px; }
th { background: #e8edf4; }
img { margin: 6px 0; }
.figure { margin-bottom: 0; -pdf-keep-with-next: true; }
.caption { font-size: 8pt; color: #4a5568; margin-top: 0; margin-bottom: 10px; text-align: left; }
.label { -pdf-keep-with-next: true; }
h1.new-page { page-break-before: always; }
"""


def resolve_link(uri: str, base_dir: Path) -> str:
    """Resolves an image path written relative to ``base_dir``; URLs pass through unchanged."""
    if uri.startswith(("http://", "https://", "data:")):
        return uri
    return str((Path(base_dir) / uri).resolve())


def figure_size(
    pixel_width: int, pixel_height: int, explicit_width: int | None = None
) -> tuple[int, int]:
    """Returns the width and height in points of a figure, keeping its aspect ratio.

    An ``explicit_width`` is used as given. Otherwise strips with a width to
    height ratio of at least ``STRIP_RATIO`` take ``FULL_WIDTH`` and other
    figures ``SINGLE_WIDTH``. The height never exceeds ``MAX_HEIGHT``; a taller
    figure is narrowed until it fits.
    """
    ratio = pixel_width / pixel_height
    width = explicit_width or (FULL_WIDTH if ratio >= STRIP_RATIO else SINGLE_WIDTH)
    if width / ratio > MAX_HEIGHT:
        return int(MAX_HEIGHT * ratio), MAX_HEIGHT
    return width, round(width / ratio)


def is_photographic(image: Image.Image) -> bool:
    """Tells photos, which compress well as JPEG, from charts and diagrams, which stay lossless.

    An image with more than ``PHOTO_COLORS`` distinct colours counts as photographic.
    """
    return len(image.convert("RGB").getcolors(maxcolors=PHOTO_COLORS) or ()) == 0


def flatten_on_white(image: Image.Image) -> Image.Image:
    """Returns an RGB copy of ``image``; any transparency is composited on a white background."""
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def embedded_copy(source: str, width_points: int, build_dir: Path) -> str:
    """Returns the path of the image to embed at ``width_points`` wide on the page.

    When the file has more pixels than ``PRINT_DPI`` needs at that width, or
    has transparency, a copy is written to ``build_dir`` (downscaled, and
    composited on white) and its path is returned. Otherwise the source path is
    returned unchanged. URLs pass through.
    """
    if source.startswith(("http://", "https://", "data:")):
        return source
    target = round(width_points * PRINT_DPI / 72)
    with Image.open(source) as image:
        transparent = image.mode in ("RGBA", "LA", "P")
        if image.width <= target and not transparent:
            return source
        flat = flatten_on_white(image)
    if flat.width > target:
        flat = flat.resize((target, max(1, round(flat.height * target / flat.width))), Image.LANCZOS)
    resized = flat
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    name = f"{sha1(source.encode()).hexdigest()[:12]}_{Path(source).stem}"
    if is_photographic(resized):
        copy = build_dir / f"{name}.jpg"
        resized.convert("RGB").save(copy, quality=JPEG_QUALITY, optimize=True)
    else:
        copy = build_dir / f"{name}.png"
        resized.save(copy, optimize=True)
    return str(copy.resolve())


def size_images(html: str, base_dir: Path, build_dir: Path) -> str:
    """Sizes every image, points it at a print-resolution copy and adds its caption.

    A width already set on the tag is honoured, in points. Widths are written in
    pixels. A paragraph holding only an image becomes a figure paragraph followed
    by a caption paragraph with the alt text.
    """

    def sized(tag: str) -> str:
        """Rewrites one image tag."""
        source = resolve_link(re.search(r'src="([^"]+)"', tag).group(1), base_dir)
        explicit = re.search(r'width="(\d+)"', tag)
        tag = re.sub(r'\s(width|height)="[^"]*"', "", tag)
        with Image.open(source) as image:
            width, height = figure_size(
                image.width, image.height, int(explicit.group(1)) if explicit else None
            )
        embedded = embedded_copy(source, width, build_dir)
        tag = re.sub(r'src="[^"]+"', f'src="{embedded}"', tag)
        pixel_width = round(width * PIXELS_PER_POINT)
        pixel_height = round(height * PIXELS_PER_POINT)
        return tag.replace("<img ", f'<img width="{pixel_width}" height="{pixel_height}" ', 1)

    def figure(match: re.Match) -> str:
        """Rewrites one paragraph that holds only an image."""
        tag = sized(match.group(1))
        alt = re.search(r'alt="([^"]*)"', tag)
        caption = f'<p class="caption">{alt.group(1)}</p>' if alt and alt.group(1) else ""
        return f'<p class="figure">{tag}</p>{caption}'

    html = re.sub(r"<p>(<img [^>]*>)</p>", figure, html)
    return re.sub(
        r"<img [^>]*>", lambda match: match.group(0) if "height=" in match.group(0) else sized(match.group(0)), html
    )


def mark_page_breaks_and_labels(html: str) -> str:
    """Marks every h1 after the first to start a page and every all-bold paragraph to stay with its block."""
    first = html.find("<h1>")
    if first != -1:
        head, tail = html[: first + 4], html[first + 4 :]
        html = head + tail.replace("<h1>", '<h1 class="new-page">')
    return re.sub(r"<p>(<strong>(?:(?!</strong>).)*</strong>)</p>", r'<p class="label">\1</p>', html)


def merge_sections(sections_dir: Path) -> str:
    """Joins every Markdown section in filename order."""
    parts = [
        path.read_text(encoding="utf-8").strip()
        for path in sorted(Path(sections_dir).glob("*.md"))
    ]
    return "\n\n".join(parts) + "\n"


def default_build_dir(sections_dir: Path) -> Path:
    """Returns the folder for downscaled copies: ``build/images`` beside the report folder."""
    return Path(sections_dir).resolve().parent.parent / "build" / "images"


def render_html(sections_dir: Path, build_dir: Path | None = None) -> str:
    """Renders the merged sections into one HTML page, with images sized from their files.

    Image paths in the sections are relative to the parent of ``sections_dir``.
    Downscaled copies go to ``build_dir``, by default ``build/images`` at the repository root.
    """
    sections_dir = Path(sections_dir)
    build_dir = Path(build_dir) if build_dir else default_build_dir(sections_dir)
    body = markdown(
        merge_sections(sections_dir),
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    body = body.replace("<table>", '<table cellspacing="0">')
    body = mark_page_breaks_and_labels(body)
    body = size_images(body, sections_dir.parent, build_dir)
    return (
        "<html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )


def build(sections_dir: Path, output_pdf: Path) -> None:
    """Renders the merged sections into one PDF file."""
    html = render_html(sections_dir)
    report_dir = Path(sections_dir).resolve().parent
    allowed_dir = report_dir.parent
    with open(output_pdf, "wb") as handle, use_policy(ResourceAccessPolicy(base_dir=allowed_dir)):
        result = pisa.CreatePDF(
            html,
            dest=handle,
            link_callback=lambda uri, _relative: resolve_link(uri, report_dir),
            encoding="utf-8",
        )
    if result.err:
        raise RuntimeError(f"PDF generation reported {result.err} error(s)")


if __name__ == "__main__":
    build(SECTIONS_DIR, OUTPUT_PDF)
    print(f"wrote {OUTPUT_PDF.relative_to(ROOT)}")
