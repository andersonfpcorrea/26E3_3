"""Tests for the report builder."""

import logging
import re
from pathlib import Path

from markdown import markdown
from PIL import Image
from pypdf import PdfReader
from pypdf.generic import ContentStream

from build_report import (
    FULL_WIDTH,
    SINGLE_WIDTH,
    PRINT_DPI,
    build,
    embedded_copy,
    figure_size,
    merge_sections,
    normalise_list_indent,
    render_html,
    resolve_link,
)


def pixels(points):
    """Converts a width in points to the pixels the PDF library expects in a width attribute."""
    return round(points * 4 / 3)


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
    assert f'width="{pixels(FULL_WIDTH)}"' in html
    narrow = render_html(
        make_report(tmp_path / "b", image_size=(1650, 600), markdown_image='\n![Figura](figures/f.png){: width="150"}\n')
    )
    assert f'width="{pixels(150)}"' in narrow


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


def test_html_widths_are_in_pixels_so_that_points_print_correctly(tmp_path):
    """A width attribute is read as pixels, so a point width is scaled by 4/3 in the HTML."""
    html = render_html(make_report(tmp_path, image_size=(1080, 600)))
    assert f'width="{pixels(SINGLE_WIDTH)}"' in html


def image_widths_in_points(pdf_path):
    """Returns the printed width in points of every image drawn in a PDF, read from the content stream."""
    widths = []
    for page in PdfReader(pdf_path).pages:
        content = ContentStream(page.get_contents(), page.pdf)
        scale = None
        for operands, operator in content.operations:
            if operator == b"cm":
                scale = float(operands[0])
            elif operator == b"Do" and scale:
                widths.append(scale)
    return widths


def test_a_full_width_figure_fills_the_text_width_in_the_pdf(tmp_path):
    """A wide strip prints at the text width of an A4 page with 1.7 cm margins, not at 75% of it."""
    sections = make_report(tmp_path, image_size=(1650, 600))
    output = tmp_path / "out.pdf"
    build(sections, output)
    (width,) = image_widths_in_points(output)
    assert abs(width - FULL_WIDTH) < 3


def test_alt_text_is_rendered_as_a_caption_after_the_image(tmp_path):
    """Each image is followed by a caption paragraph holding its alt text."""
    sections = make_report(
        tmp_path, image_size=(1650, 600), markdown_image="\n![Linha grossa: média](figures/f.png)\n"
    )
    html = render_html(sections)
    assert html.index("<img ") < html.index('<p class="caption">Linha grossa: média</p>')
    output = tmp_path / "out.pdf"
    build(sections, output)
    text = "".join(page.extract_text() for page in PdfReader(output).pages)
    assert "Linha grossa: média" in text


def test_images_stay_with_their_captions(tmp_path):
    """The image paragraph is kept with the caption that follows it."""
    html = render_html(make_report(tmp_path, image_size=(1650, 600)))
    assert '<p class="figure"><img ' in html
    assert ".figure {" in html and "-pdf-keep-with-next" in html


def test_each_top_level_heading_after_the_first_starts_a_page(tmp_path):
    """Sections are not separated by a rule, and every h1 but the first breaks the page."""
    sections = tmp_path / "report" / "secoes"
    sections.mkdir(parents=True)
    (sections / "00_a.md").write_text("# Primeira\n\ntexto\n")
    (sections / "10_b.md").write_text("# Segunda\n\ntexto\n")
    html = render_html(sections)
    assert "<hr" not in html
    assert html.count('<h1 class="new-page">') == 1
    assert html.index("Primeira") < html.index('<h1 class="new-page">Segunda')
    output = tmp_path / "out.pdf"
    build(sections, output)
    pages = [page.extract_text() for page in PdfReader(output).pages]
    assert len(pages) == 2
    assert "Segunda" in pages[1] and "Segunda" not in pages[0]


def test_bold_label_paragraphs_stay_with_the_block_that_follows(tmp_path):
    """A paragraph made only of bold text introduces a block and is kept with it."""
    sections = tmp_path / "report" / "secoes"
    sections.mkdir(parents=True)
    (sections / "00_a.md").write_text("# T\n\n**Rótulo**\n\n- item\n\nTexto com **negrito** no meio.\n")
    html = render_html(sections)
    assert '<p class="label"><strong>Rótulo</strong></p>' in html
    assert "<p>Texto com <strong>negrito</strong>" in html


def test_transparent_images_are_flattened_on_white(tmp_path):
    """A transparent image is composited on white before it is downscaled, as JPEG or PNG."""
    source = tmp_path / "t.png"
    Image.new("RGBA", (3000, 1000), (0, 0, 0, 0)).save(source)
    copy = embedded_copy(str(source), 300, tmp_path / "build")
    with Image.open(copy) as image:
        assert image.convert("RGB").getpixel((10, 10)) == (255, 255, 255)
    small = tmp_path / "s.png"
    Image.new("RGBA", (100, 100), (0, 0, 0, 0)).save(small)
    copy = embedded_copy(str(small), 300, tmp_path / "build")
    with Image.open(copy) as image:
        assert image.convert("RGB").getpixel((10, 10)) == (255, 255, 255)


def test_two_space_nested_bullets_are_rendered_as_nested_lists():
    """Nested bullets indented by two spaces, as Prettier writes them, still nest in the HTML."""
    html = markdown(normalise_list_indent("- a\n  - nested\n- b\n"), extensions=["sane_lists"])
    assert "<li>a<ul>\n<li>nested</li>" in html


def test_three_space_nested_items_under_numbered_items_are_rendered_as_nested_lists():
    """Items indented by three spaces under a numbered item nest in the HTML."""
    html = markdown(normalise_list_indent("1. a\n   - nested\n2. b\n"), extensions=["sane_lists"])
    assert "<li>a<ul>\n<li>nested</li>" in html


def test_two_space_indentation_scales_every_level_of_a_list():
    """A second nesting level written with four spaces in a two-space list becomes eight spaces."""
    source = "- a\n  - b\n    - c\n"
    assert normalise_list_indent(source) == "- a\n    - b\n        - c\n"


def test_four_space_lists_are_left_unchanged():
    """Lists already indented by four spaces are returned exactly as given."""
    source = "- a\n    - nested\n- b\n\nTexto.\n"
    assert normalise_list_indent(source) == source


def test_fenced_code_keeps_its_indentation():
    """Lines inside a fenced code block are never re-indented."""
    source = "- a\n  - b\n\n```\n  - not a list\n```\n"
    assert normalise_list_indent(source).endswith("```\n  - not a list\n```\n")


def test_rendered_report_nests_two_space_bullets(tmp_path):
    """The report renderer applies the indentation fix to the merged sections."""
    (tmp_path / "secoes").mkdir()
    (tmp_path / "secoes" / "00_a.md").write_text("# T\n\n- a\n  - nested\n", encoding="utf-8")
    html = render_html(tmp_path / "secoes", tmp_path / "build")
    assert "<li>a<ul>" in html
