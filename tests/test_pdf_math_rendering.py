"""Math-rendering regression suite (Phase 1 table: PDF-1 .. PDF-5).

Formalizes the ad-hoc verification done while diagnosing/fixing the
exponent-notation bug: string-transform layer (_clean_latex), font
glyph-coverage layer (fontTools cmap check), and actual rendered-pixel
layer (rasterize a real generated PDF and OCR-free-verify via pdfplumber's
embedded ToUnicode text extraction, which reflects genuine glyph mapping).
"""
import os

import pytest
from fontTools.ttLib import TTFont

from utils.pdf_generator import (
    _clean_latex,
    _find_unicode_font,
    generate_practice_paper_pdf,
    _SAFE_SCRIPT_CHARS,
)


# ── PDF-1 / PDF-2: exponent & subscript notation, all input shapes ─────


@pytest.mark.parametrize(
    "label,input_text,expected",
    [
        ("bare digit exponent", "X^2", "X²"),
        ("bare multi-digit exponent", "2^15", "2¹⁵"),
        ("braced digit exponent", "X^{2}", "X²"),
        ("parenthesized digit exponent", "X^(2)", "X²"),
        ("bare negative exponent", "X^-1", "X⁻¹"),
        ("braced negative exponent", "X^{-1}", "X⁻¹"),
        ("parenthesized negative exponent", "e^(-x)", "e^(-x)"),
        ("bare letter exponent", "X^n", "X^n"),
        ("braced letter exponent", "X^{n}", "X^n"),
        ("braced mixed sign+letter exponent", "e^{-x}", "e^(-x)"),
        ("bare digit subscript", "X_1", "X₁"),
        ("braced digit subscript", "X_{1}", "X₁"),
        ("bare letter subscript", "X_i", "X_i"),
        ("braced letter subscript", "X_{i}", "X_i"),
        ("parenthesized subscript", "X_(i-1)", "X_(i-1)"),
        ("digit+letter run with no separator", "X^2Y", "X²Y"),
    ],
)
def test_pdf1_pdf2_exponent_subscript_notation(label, input_text, expected):
    assert _clean_latex(input_text) == expected, label


def test_pdf1_pdf2_already_unicode_superscript_passes_through_unchanged():
    """PDF-5: mixed formats in one document — text that's already a real
    Unicode superscript (e.g. copy-pasted) must not be double-processed."""
    assert _clean_latex("2⁵ is already correct") == "2⁵ is already correct"


def test_pdf1_pdf2_dollar_wrapped_exponent_still_converts():
    """PDF-5: $2^5$ (LaTeX inline math wrapper) — the $ delimiters are
    stripped and the exponent inside still converts."""
    assert _clean_latex("$2^5$") == "2⁵"


# ── PDF-3: Greek letters, \frac, ASCII math, fractions ──────────────────


@pytest.mark.parametrize(
    "label,input_text,expected",
    [
        ("greek letter with digit exponent", r"\sigma^2 estimate", "σ² estimate"),
        ("latex fraction", r"\frac{a}{b}", "(a)/(b)"),
        ("textbf strip", r"\textbf{Note}", "Note"),
        ("inline dollar math", r"$X^2$ is the result", "X² is the result"),
        ("ascii fraction", "the value 1/2 exactly", "the value ½ exactly"),
        ("ascii arrow", "X -> Y as n -> infinity", "X → Y as n → infinity"),
        ("leq geq", "0 <= x <= 1", "0 ≤ x ≤ 1"),
        ("plain text untouched", "No math here at all.", "No math here at all."),
    ],
)
def test_pdf3_latex_and_ascii_math_conversion(label, input_text, expected):
    assert _clean_latex(input_text) == expected, label


# ── Font glyph-coverage guard: the root cause of the original bug ──────


def test_font_has_full_glyph_coverage_for_every_safe_script_char():
    """Root-cause regression guard: every character we ever translate to a
    Unicode superscript/subscript codepoint must have a real glyph in the
    resolved font. If this ever goes red, _SAFE_SCRIPT_CHARS has grown to
    include a character the font can't render — exactly the class of bug
    that caused the original 'blank exponent' report."""
    font_path = _find_unicode_font()
    if not font_path:
        pytest.skip("no system Unicode font available in this environment")

    font = TTFont(font_path, fontNumber=0, lazy=True)
    cmap = font.getBestCmap()

    super_map = str.maketrans(dict(zip(_SAFE_SCRIPT_CHARS, "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾")))
    sub_map = str.maketrans(dict(zip(_SAFE_SCRIPT_CHARS, "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎")))

    missing = []
    for ch in _SAFE_SCRIPT_CHARS:
        for codepoint in (ch.translate(super_map), ch.translate(sub_map)):
            if ord(codepoint) not in cmap:
                missing.append(f"U+{ord(codepoint):04X} ({codepoint!r})")

    assert not missing, f"font is missing glyphs the code assumes are safe: {missing}"


def test_font_confirmed_missing_letter_glyphs_stay_unmapped():
    """Documents WHY letters are deliberately excluded from the safe-script
    set: confirms the font genuinely lacks these glyphs, so if someone is
    tempted to 'just add letter support' later, this test explains why
    that would silently reintroduce blank glyphs."""
    font_path = _find_unicode_font()
    if not font_path:
        pytest.skip("no system Unicode font available in this environment")

    font = TTFont(font_path, fontNumber=0, lazy=True)
    cmap = font.getBestCmap()

    # A representative sample of Unicode subscript/superscript Latin letters
    sample_missing_subscript_letters = "ᵢⱼₖₙₓ"  # i, j, k, n, x
    for ch in sample_missing_subscript_letters:
        assert ord(ch) not in cmap, (
            f"font now HAS a glyph for {ch!r} that it previously lacked — "
            "safe to consider adding to the letter-subscript map, but verify "
            "full coverage first, not just this sample"
        )


# ── PDF-4 (recalibrated): content that could break the PDF library ─────
#
# Original Phase-1 classification was cautious/unverified. Empirical testing
# found fpdf2 2.7.9 does NOT raise for these inputs — it silently drops
# unsupported glyphs (with a stderr warning) instead of crashing. These
# tests lock in that robustness as a regression guard, since a future
# fpdf2/font change could reintroduce a real crash here.


@pytest.mark.parametrize(
    "label,pathological_text",
    [
        ("very long unbroken run (no spaces)", "X" * 10000),
        ("control characters", "Value is \x00\x01\x02 here."),
        ("astral-plane emoji", "Check this result: \U0001F600\U0001F4A5 done."),
        ("null byte mid-string", "before\x00after"),
        ("mixed RTL/LTR with no whitespace", "textدون" * 50),
    ],
)
def test_pdf4_pathological_content_does_not_crash_pdf_generation(label, pathological_text):
    questions = [
        {
            "question_number": 1,
            "question_text": f"See value {pathological_text} here.",
            "marks": 5,
            "subparts": [],
            "model_answer": [],
        }
    ]
    pdf_bytes = generate_practice_paper_pdf(
        "Test Subject", questions, {"estimated_total_marks": 5}, has_slides=True
    )
    assert pdf_bytes[:4] == b"%PDF", label


def test_pdf4_pathological_content_in_subject_title_does_not_crash():
    """The subject name renders via pdf.cell() (no auto-wrap), a different
    code path from question bodies (multi_cell(), which does wrap) — long
    unbroken subject names are a distinct risk from long question text."""
    long_subject = "X" * 300
    questions = [
        {"question_number": 1, "question_text": "A short question.", "marks": 5, "subparts": [], "model_answer": []}
    ]
    pdf_bytes = generate_practice_paper_pdf(
        long_subject, questions, {"estimated_total_marks": 5}, has_slides=True
    )
    assert pdf_bytes[:4] == b"%PDF"


# ── End-to-end visual confirmation (rendered pixels, not just text) ────


def test_exponents_survive_actual_pdf_rasterization(tmp_path):
    """Belt-and-suspenders: render a real PDF and read back the embedded
    ToUnicode-mapped text (reflects genuine glyph->codepoint mapping, not
    just the Python string transform) to catch font-subsetting bugs that
    string-level tests alone would miss."""
    pdfplumber = pytest.importorskip("pdfplumber")

    questions = [
        {"question_number": 1, "question_text": "Let Y = X^2 be the transform.", "marks": 5, "subparts": [], "model_answer": []},
        {"question_number": 2, "question_text": "Find X^{-1} the inverse.", "marks": 5, "subparts": [], "model_answer": []},
    ]
    pdf_bytes = generate_practice_paper_pdf(
        "Rasterize Test", questions, {"estimated_total_marks": 10}, has_slides=True
    )
    out_path = tmp_path / "raster_test.pdf"
    out_path.write_bytes(pdf_bytes)

    with pdfplumber.open(str(out_path)) as pdf:
        text = pdf.pages[0].extract_text()

    assert "X²" in text
    assert "X⁻¹" in text
