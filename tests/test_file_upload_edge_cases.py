"""Edge cases for file upload / ingest (Phase 1 table: FU-1 .. FU-9).

Each test is tagged with its Phase-1 case ID in the docstring so coverage
maps directly back to the audit table. Where the underlying pipeline
function has no local safety net but a pipeline-level `except Exception`
catches it anyway, both layers are tested separately so the gap is visible
rather than hidden by the outer catch.
"""
import os
import tempfile

import pytest

from utils.pdf_utils import extract_text_from_pdf, extract_text_from_pptx
from pipeline.ingest import ingest_past_paper, ingest_slides
from pipeline.graph import run_pipeline

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


def _write_temp(suffix: str, content: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(content)
        return f.name


# ── FU-1: non-PDF renamed to .pdf ───────────────────────────────────────


def test_fu1_non_pdf_raises_at_extract_layer():
    """FU-1: extract_text_from_pdf has no defensive check on a fake PDF."""
    path = _write_temp(".pdf", b"This is definitely not a PDF, just text.")
    with pytest.raises(Exception):
        extract_text_from_pdf(path)


def test_fu1_non_pdf_caught_by_pipeline(isolated_chroma):
    """FU-1: run_pipeline must not crash the app; it should report an error."""
    path = _write_temp(".pdf", b"This is definitely not a PDF, just text.")
    result = run_pipeline("Edge Case Subject", None, {"2099": path})
    assert result.get("error"), "pipeline should surface a graceful error, not raise"


# ── FU-2: 0-byte file ────────────────────────────────────────────────────


def test_fu2_zero_byte_pdf_raises_at_extract_layer():
    """FU-2: a 0-byte file is not a parseable PDF."""
    path = _write_temp(".pdf", b"")
    with pytest.raises(Exception):
        extract_text_from_pdf(path)


def test_fu2_zero_byte_pdf_caught_by_pipeline(isolated_chroma):
    path = _write_temp(".pdf", b"")
    result = run_pipeline("Edge Case Subject", None, {"2099": path})
    assert result.get("error")


# ── FU-3: corrupt / truncated PDF ───────────────────────────────────────


def test_fu3_truncated_pdf_raises_at_extract_layer():
    """FU-3: a real PDF cut off mid-body is a distinct failure mode from
    FU-1 (garbage bytes) — pdfminer raises PSEOF rather than PDFSyntaxError."""
    with open(os.path.join(FIXTURES, "paper_2022.pdf"), "rb") as f:
        real_bytes = f.read()
    truncated = real_bytes[: len(real_bytes) // 3]
    path = _write_temp(".pdf", truncated)
    with pytest.raises(Exception):
        extract_text_from_pdf(path)


def test_fu3_truncated_pdf_caught_by_pipeline(isolated_chroma):
    with open(os.path.join(FIXTURES, "paper_2022.pdf"), "rb") as f:
        real_bytes = f.read()
    truncated = real_bytes[: len(real_bytes) // 3]
    path = _write_temp(".pdf", truncated)
    result = run_pipeline("Edge Case Subject", None, {"2099": path})
    assert result.get("error")


# ── FU-4: password-protected / encrypted PDF ────────────────────────────


def test_fu4_encrypted_pdf_raises_at_extract_layer():
    path = os.path.join(FIXTURES, "encrypted.pdf")
    with pytest.raises(Exception):
        extract_text_from_pdf(path)


def test_fu4_encrypted_pdf_caught_by_pipeline(isolated_chroma):
    path = os.path.join(FIXTURES, "encrypted.pdf")
    result = run_pipeline("Edge Case Subject", None, {"2099": path})
    assert result.get("error")


# ── FU-5: scanned / image-only PDF (zero extractable text) ─────────────


def test_fu5_scanned_pdf_yields_no_pages():
    """Confirms the input shape: pdfplumber finds no text layer at all."""
    path = os.path.join(FIXTURES, "scanned_no_text.pdf")
    pages = extract_text_from_pdf(path)
    assert pages == []


def test_fu5_scanned_pdf_does_not_crash_the_pipeline(isolated_chroma):
    """FU-5: this is the real MUST-FIX finding — ingest_past_paper crashes
    on empty content (sentence-transformers .encode([]) raises IndexError,
    verified directly against this exact input). The pipeline-level catch
    in node_ingest absorbs it, so the app itself must not crash."""
    path = os.path.join(FIXTURES, "scanned_no_text.pdf")
    result = run_pipeline("Edge Case Subject", None, {"2099": path})
    assert result.get("error"), "pipeline should report an error, not raise uncaught"


def test_fu5_scanned_pdf_error_message_is_actionable(isolated_chroma):
    """Stricter than the test above: the CURRENT error message is the raw
    'list index out of range' from sentence-transformers, which tells a
    user nothing about *why* (a scanned/image-only PDF has no text layer).
    This is expected to FAIL today — it documents a real, verified UX gap
    that the crash-catch alone doesn't fix."""
    path = os.path.join(FIXTURES, "scanned_no_text.pdf")
    result = run_pipeline("Edge Case Subject", None, {"2099": path})
    msg = (result.get("error") or "").lower()
    assert any(kw in msg for kw in ("text", "scan", "empty", "no content")), (
        f"error message is not actionable for this failure mode: {result.get('error')!r}"
    )


def test_fu5_ingest_past_paper_raises_directly(isolated_chroma):
    """Same finding as above, isolated to the exact function/line that
    actually breaks. ingest_past_paper now guards the zero-chunk case and
    raises an actionable ValueError instead of letting sentence-transformers'
    IndexError leak out uncaught."""
    path = os.path.join(FIXTURES, "scanned_no_text.pdf")
    with pytest.raises(ValueError, match="(?i)text|scan|empty|no content"):
        ingest_past_paper(path, "Edge Case Subject Direct", "2099")


# ── FU-6: corrupt .pptx slides ──────────────────────────────────────────


def test_fu6_corrupt_pptx_raises_at_extract_layer():
    path = _write_temp(".pptx", b"not a real pptx zip archive at all")
    with pytest.raises(Exception):
        extract_text_from_pptx(path)


def test_fu6_corrupt_pptx_caught_by_pipeline(isolated_chroma):
    slides_path = _write_temp(".pptx", b"not a real pptx zip archive at all")
    paper_path = os.path.join(FIXTURES, "paper_2022.pdf")
    result = run_pipeline("Edge Case Subject", slides_path, {"2022": paper_path})
    assert result.get("error")


# ── FU-9: single paper (below the recommended 3-5) ──────────────────────


def test_fu9_single_paper_ingest_succeeds(isolated_chroma):
    """FU-9: not a bug — just confirms ingest doesn't require a minimum
    paper count to function."""
    path = os.path.join(FIXTURES, "paper_2022.pdf")
    n_chunks = ingest_past_paper(path, "Single Paper Subject", "2022")
    assert n_chunks > 0
