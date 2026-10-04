"""Tests for the report builder."""

import logging
import re
from pathlib import Path

from PIL import Image
from pypdf import PdfReader

from build_report import (
    FULL_WIDTH,
    SINGLE_WIDTH,
    PRINT_DPI,
    build,
    figure_size,
    merge_sections,
    render_html,
    resolve_link,
)


def make_report(tmp_path, image_size=None, markdown_image=None):
    """Writes a report folder with one section and, optionally, one figure; returns the sections folder."""
    sections = tmp_path / "report" / "secoes"
    sections.mkdir(parents=True)
    body = "# Relatório\n\n- Acurácia de validação\n\n| Classe | Acurácia |\n| --- | --- |\n| bike | 0,98 |\n"
    if image_size:
        figures = tmp_path / "report" / "figures"
        figures.mkdir()
        Image.new("RGB", image_size, (200, 30, 30)).save(figures / "f.png")
        body += markdown_image or "\n![Figura](figures/f.png)\n"
    (sections / "00_a.md").write_text(body)
    return sections


def test_merge_sections_follows_filename_order(tmp_path):
    """Sections are joined by filename, whatever order they were written in."""
    (tmp_path / "30_c.md").write_text("# Terceira\n")
    (tmp_path / "00_a.md").write_text("# Primeira\n")
    merged = merge_sections(tmp_path)
    assert merged.index("Primeira") < merged.index("Terceira")


def test_resolve_link_uses_the_folder_it_is_given(tmp_path):
    """Relative image paths resolve against the folder passed in, and URLs pass through."""
    assert resolve_link("figures/a.png", tmp_path) == str((tmp_path / "figures" / "a.png").resolve())
    assert resolve_link("https://example.org/a.png", tmp_path) == "https://example.org/a.png"


def test_wide_figures_take_the_full_width_and_keep_their_aspect_ratio():
    """A multi-panel strip uses the full text width."""
    width, height = figure_size(1650, 600)
    assert width == FULL_WIDTH
    assert height == round(FULL_WIDTH * 600 / 1650)


def test_single_charts_take_a_narrower_width():
    """A chart that is not a wide strip is drawn at about two thirds of the text width."""
    assert figure_size(1080, 600)[0] == SINGLE_WIDTH
    assert figure_size(840, 840)[0] == SINGLE_WIDTH
    assert SINGLE_WIDTH < FULL_WIDTH


def test_tall_figures_are_limited_by_page_height_with_the_ratio_kept():
    """A tall sheet shrinks so that it fits on one page."""
    width, height = figure_size(1350, 3206)
    assert height <= 700
    assert abs(width / height - 1350 / 3206) < 0.01


def test_an_explicit_width_wins_over_the_aspect_ratio_rule():
    """A width given in the Markdown image line is used as written, with the ratio kept."""
    assert figure_size(840, 840, explicit_width=FULL_WIDTH) == (FULL_WIDTH, FULL_WIDTH)


def test_render_html_sizes_images_from_their_files(tmp_path):
    """Every image gets width and height from its file, and an explicit width in the line is honoured."""
    html = render_html(make_report(tmp_path, image_size=(1650, 600)))
    assert f'width="{FULL_WIDTH}"' in html
    narrow = render_html(
        make_report(tmp_path / "b", image_size=(1650, 600), markdown_image='\n![Figura](figures/f.png){: width="200"}\n')
    )
    assert 'width="200"' in narrow


def test_headings_stay_with_the_content_that_follows(tmp_path):
    """The style sheet keeps every heading level on the page of the next block."""
    sections = tmp_path / "report" / "secoes"
    sections.mkdir(parents=True)
    (sections / "00_a.md").write_text("# T\n\n## S\n\n### U\n\ntexto\n")
    html = render_html(sections)
    for level in ("h1", "h2", "h3"):
        assert f"{level} {{" in html
    assert html.count("-pdf-keep-with-next: true") >= 3


def test_build_prints_no_warning_and_keeps_accents_and_table_text(tmp_path, caplog):
    """The PDF is built without warnings and its text keeps the accents of heading, list and table."""
    sections = make_report(tmp_path, image_size=(840, 840))
    output = tmp_path / "out.pdf"
    with caplog.at_level(logging.WARNING):
        build(sections, output)
    assert not [record for record in caplog.records if record.levelno >= logging.WARNING]
    text = "".join(page.extract_text() for page in PdfReader(output).pages)
    assert "Relatório" in text
    assert "Acurácia de validação" in text
    assert "Acurácia" in text and "0,98" in text


def embedded_sources(html):
    """Returns the image paths written in the ``src`` attributes of an HTML page."""
    return re.findall(r'<img [^>]*src="([^"]+)"', html)


def test_large_images_are_embedded_smaller_than_their_source(tmp_path):
    """A figure much larger than its printed size is replaced by a downscaled copy in the build folder."""
    sections = make_report(tmp_path, image_size=(3000, 1000))
    build_dir = tmp_path / "build"
    html = render_html(sections, build_dir=build_dir)
    (source,) = embedded_sources(html)
    assert Path(source).parent == build_dir.resolve()
    with Image.open(source) as copy:
        assert copy.width <= round(FULL_WIDTH * PRINT_DPI / 72)
        assert copy.width < 3000
        assert abs(copy.width / copy.height - 3) < 0.05


def test_small_images_are_not_upscaled(tmp_path):
    """A figure already below the print resolution keeps its pixels."""
    sections = make_report(tmp_path, image_size=(300, 200))
    html = render_html(sections, build_dir=tmp_path / "build")
    (source,) = embedded_sources(html)
    with Image.open(source) as copy:
        assert copy.size == (300, 200)


def test_source_images_are_left_untouched(tmp_path):
    """Downscaling never rewrites the file under report/figures."""
    sections = make_report(tmp_path, image_size=(3000, 1000))
    figure = tmp_path / "report" / "figures" / "f.png"
    before = figure.read_bytes()
    render_html(sections, build_dir=tmp_path / "build")
    assert figure.read_bytes() == before


def test_photographs_are_embedded_as_jpeg_and_flat_charts_stay_png(tmp_path):
    """Images with many colours become JPEG copies; flat-colour charts keep lossless PNG."""
    from numpy.random import default_rng

    sections = make_report(tmp_path, image_size=(3000, 1000))
    figure = tmp_path / "report" / "figures" / "f.png"
    noise = default_rng(0).integers(0, 256, (1000, 3000, 3), dtype="uint8")
    Image.fromarray(noise).save(figure)
    (photo,) = embedded_sources(render_html(sections, build_dir=tmp_path / "b1"))
    assert photo.endswith(".jpg")
    Image.new("RGB", (3000, 1000), (10, 20, 30)).save(figure)
    (chart,) = embedded_sources(render_html(sections, build_dir=tmp_path / "b2"))
    assert chart.endswith(".png")
