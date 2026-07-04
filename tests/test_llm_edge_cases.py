"""LLM generation edge cases (Phase 1 table: LLM-1 .. LLM-8).

All Gemini calls are mocked via conftest.py's mock_gemini_* helpers — these
tests never touch the real API, so they're free and deterministic to run.
"""
import pytest

from pipeline.style_extractor import extract_paper_style
from pipeline.analyzer import (
    build_topic_frequency_map_no_slides,
    enrich_topics_with_concepts,
)
from pipeline.generator import _parse_structured_output
from pipeline.graph import run_pipeline
from utils.pdf_generator import generate_practice_paper_pdf

from conftest import mock_gemini_text, mock_gemini_raises, mock_gemini_blocked_response

MALFORMED_JSON = "this is not valid json at all {{{"


def _seed_papers_collection(client, subject_key: str):
    try:
        client.delete_collection(f"papers_{subject_key}")
    except Exception:
        pass
    col = client.create_collection(f"papers_{subject_key}")
    col.add(
        documents=["Q1: Find the mean of X.", "Q2: State the variance formula."],
        embeddings=[[0.1] * 384, [0.2] * 384],
        ids=["q1", "q2"],
        metadatas=[{"year": "2022", "question_num": "1"}, {"year": "2022", "question_num": "2"}],
    )


# ── LLM-1: style_extractor has no local safety net for bad JSON ────────


def test_llm1_style_extractor_raises_on_malformed_json(isolated_chroma):
    """LLM-1: unlike the other two Gemini JSON call sites, extract_paper_style
    has no try/except at all around json.loads(). Expected to raise directly
    — this documents that a single Gemini formatting hiccup aborts the whole
    pipeline run instead of falling back to sane defaults."""
    _seed_papers_collection(isolated_chroma, "llm1_subject")
    with mock_gemini_text(MALFORMED_JSON):
        with pytest.raises(Exception):
            extract_paper_style("llm1_subject")


def test_llm1_caught_gracefully_one_layer_up(isolated_chroma):
    """Same malformed-JSON case, but through the full pipeline — node-level
    except means the app itself must not crash even though the function
    above has no local fallback."""
    _seed_papers_collection(isolated_chroma, "llm1_subject_pipeline")
    with mock_gemini_text(MALFORMED_JSON):
        result = run_pipeline(
            "llm1 subject pipeline", None, {"2022": "tests/fixtures/paper_2022.pdf"}
        )
    assert result.get("error")


# ── LLM-2: enrich_topics_with_concepts already has a graceful fallback ──


def test_llm2_malformed_json_falls_back_to_empty_concepts():
    """LLM-2: already handled — locks in the existing graceful fallback."""
    topics = [{"preview": "Some slide text", "sample_questions": ["Q1 text"]}]
    with mock_gemini_text(MALFORMED_JSON):
        result = enrich_topics_with_concepts(topics, "Any Subject")
    assert result[0]["concepts"] == []


# ── LLM-3: build_topic_frequency_map_no_slides — split behavior ────────


def test_llm3_generate_content_failure_raises_locally(isolated_chroma):
    """LLM-3: the generate_content() call itself is NOT wrapped locally —
    only the JSON-parsing step has a try/except. A network/API failure
    (distinct from a malformed response) propagates directly."""
    _seed_papers_collection(isolated_chroma, "llm3_subject")
    with mock_gemini_raises(Exception("503 Service Unavailable")):
        with pytest.raises(Exception):
            build_topic_frequency_map_no_slides("llm3_subject")


def test_llm3_malformed_json_has_local_fallback(isolated_chroma):
    """LLM-3b: contrast case — malformed JSON (as opposed to a failed call)
    IS caught locally with a single catch-all topic fallback. Already
    handled, locking it in."""
    _seed_papers_collection(isolated_chroma, "llm3b_subject")
    with mock_gemini_text(MALFORMED_JSON):
        result = build_topic_frequency_map_no_slides("llm3b_subject")
    assert len(result) == 1  # the catch-all fallback topic


# ── LLM-4: Gemini 429 quota error ───────────────────────────────────────


def test_llm4_quota_error_caught_gracefully_by_pipeline(isolated_chroma):
    with mock_gemini_raises(Exception("429 You exceeded your current quota")):
        result = run_pipeline(
            "llm4 quota subject", None, {"2022": "tests/fixtures/paper_2022.pdf"}
        )
    assert result.get("error")
    assert "429" in result["error"] or "quota" in result["error"].lower()


# ── LLM-5: Gemini safety-filter block ───────────────────────────────────


def test_llm5_safety_block_caught_gracefully_by_pipeline(isolated_chroma):
    with mock_gemini_blocked_response():
        result = run_pipeline(
            "llm5 safety subject", None, {"2022": "tests/fixtures/paper_2022.pdf"}
        )
    assert result.get("error")


# ── LLM-6: malformed Question/SubPart fields (wrong type) ──────────────


def test_llm6_wrong_marks_type_falls_back_gracefully():
    """LLM-6: already handled by Pydantic ValidationError -> fallback
    wrapper in _parse_structured_output. Locks in existing behavior."""
    bad_json = (
        '{"subject": "Test", "total_marks": 100, "time_allowed": "1 hour", '
        '"questions": [{"question_number": 1, "question_text": "Q", "marks": "fifteen"}]}'
    )
    result = _parse_structured_output(bad_json, {"estimated_total_marks": 100}, "Test")
    assert len(result) == 1
    assert result[0]["question_number"] == 1  # fallback pseudo-question shape


# ── LLM-7: negative marks (valid type, semantically wrong) ─────────────


def test_llm7_negative_marks_are_accepted_uncritically():
    """LLM-7: not a crash — Pydantic's `marks: int` has no range constraint,
    so a negative value passes straight through. Documents current
    (cosmetic-severity) behavior rather than asserting it's correct."""
    json_str = (
        '{"subject": "Test", "total_marks": 100, "time_allowed": "1 hour", '
        '"questions": [{"question_number": 1, "question_text": "Q", "marks": -5}]}'
    )
    result = _parse_structured_output(json_str, {"estimated_total_marks": 100}, "Test")
    assert result[0]["marks"] == -5


# ── LLM-8: empty questions list ─────────────────────────────────────────


def test_llm8_empty_questions_list_produces_valid_pdf():
    pdf_bytes = generate_practice_paper_pdf(
        "Test Subject", [], {"estimated_total_marks": 0}, has_slides=True
    )
    assert len(pdf_bytes) > 0
    assert pdf_bytes[:4] == b"%PDF"
