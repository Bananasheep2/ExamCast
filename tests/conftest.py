"""Inject test-facing helpers into the global namespace so tests can call
them by bare name (e.g. ``parse_pdf``, ``flag_duplicate_questions``) without
explicit imports."""
import builtins
import sys
import os

# Ensure the project root is on sys.path so we can import from utils/ and pipeline/
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.pdf_utils import (
    parse_pdf,
    flag_duplicate_questions,
    extract_format,
)
from pipeline.analyzer import (
    MOCK_CLASSIFIED,
    rank_question_types,
)

builtins.parse_pdf = parse_pdf
builtins.flag_duplicate_questions = flag_duplicate_questions
builtins.extract_format = extract_format
builtins.MOCK_CLASSIFIED = MOCK_CLASSIFIED
builtins.rank_question_types = rank_question_types


# ── Shared fixtures for edge-case / security / concurrency tests ───────────
#
# These are additive: nothing above this line is touched, and no existing
# test relies on anything below.

import glob
import inspect

import pytest
import chromadb
from unittest.mock import patch, MagicMock


# ── Skip fixture-dependent tests when the sample PDFs aren't present ────────
#
# The sample PDFs under tests/fixtures/ are intentionally NOT committed to the
# public repo (*.pdf is gitignored — they may be copyrighted). On a fresh clone
# they're absent, so any test that reads one would ERROR. This hook skips only
# those tests (detected by a fixtures reference in the test's own body) when the
# PDFs are missing. On a machine that HAS the fixtures, nothing is skipped and
# behavior is identical to before.

_FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _fixtures_present() -> bool:
    return bool(glob.glob(os.path.join(_FIXTURES_DIR, "*.pdf")))


def pytest_collection_modifyitems(config, items):
    if _fixtures_present():
        return
    skip_marker = pytest.mark.skip(
        reason="sample PDFs not present (tests/fixtures/*.pdf are excluded from "
        "the public repo); add your own to run these tests"
    )
    for item in items:
        func = getattr(item, "function", None)
        if func is None:
            continue
        try:
            src = inspect.getsource(func)
        except (OSError, TypeError):
            continue
        if "fixtures/" in src or "FIXTURES" in src:
            item.add_marker(skip_marker)

import pipeline.ingest as _ingest_mod
import pipeline.analyzer as _analyzer_mod
import pipeline.style_extractor as _style_mod
import pipeline.generator as _generator_mod

# Every module that does `from pipeline.ingest import get_chroma_client`
# gets its OWN name binding — patching pipeline.ingest.get_chroma_client
# alone would not affect the already-bound references in the other three
# modules, so each is patched explicitly.
_CHROMA_CLIENT_HOLDERS = [_ingest_mod, _analyzer_mod, _style_mod, _generator_mod]


@pytest.fixture
def isolated_chroma(tmp_path, monkeypatch):
    """Point every pipeline module's get_chroma_client() at a throwaway
    on-disk collection instead of the real ./chroma_db, so tests never read
    or corrupt real user data. Returns the client for direct inspection."""
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma_test"))

    def _get_client():
        return client

    for mod in _CHROMA_CLIENT_HOLDERS:
        monkeypatch.setattr(mod, "get_chroma_client", _get_client)

    return client


def mock_gemini_text(response_text: str):
    """Patch google.generativeai.GenerativeModel so every .generate_content()
    call returns a fixed text response, regardless of which pipeline module
    calls it (all modules share the same underlying `genai` module object,
    so a single patch on the attribute covers every call site)."""
    mock_response = MagicMock()
    mock_response.text = response_text
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response
    mock_cls = MagicMock(return_value=mock_model)
    return patch("google.generativeai.GenerativeModel", mock_cls)


def mock_gemini_raises(exc: Exception):
    """Patch google.generativeai.GenerativeModel so .generate_content()
    raises the given exception (simulates quota errors, network failures,
    safety-filter blocks, etc.)."""
    mock_model = MagicMock()
    mock_model.generate_content.side_effect = exc
    mock_cls = MagicMock(return_value=mock_model)
    return patch("google.generativeai.GenerativeModel", mock_cls)


def mock_gemini_blocked_response():
    """Patch google.generativeai.GenerativeModel so .generate_content()
    succeeds but accessing .text raises — this is how the real SDK behaves
    when a response is blocked by a safety filter (no valid candidate)."""
    mock_response = MagicMock()
    type(mock_response).text = property(
        lambda self: (_ for _ in ()).throw(ValueError("no valid candidate (safety block)"))
    )
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response
    mock_cls = MagicMock(return_value=mock_model)
    return patch("google.generativeai.GenerativeModel", mock_cls)
