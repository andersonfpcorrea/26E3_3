"""Tests for the report builder."""

from build_report import build, merge_sections


def test_merge_sections_follows_filename_order(tmp_path):
    """Sections are joined by filename, whatever order they were written in."""
    (tmp_path / "30_c.md").write_text("# Terceira\n")
    (tmp_path / "00_a.md").write_text("# Primeira\n")
    merged = merge_sections(tmp_path)
    assert merged.index("Primeira") < merged.index("Terceira")


def test_build_writes_pdf_with_accents_and_table(tmp_path):
    """The PDF is produced from text with accents and a table."""
    (tmp_path / "00_a.md").write_text(
        "# Relatório\n\n- Acurácia de validação\n\n| Classe | Acurácia |\n| --- | --- |\n| bike | 0,98 |\n"
    )
    output = tmp_path / "out.pdf"
    build(tmp_path, output)
    assert output.read_bytes().startswith(b"%PDF")
    assert output.stat().st_size > 1000
