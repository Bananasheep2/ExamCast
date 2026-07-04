"""UI-level tests driven via app.py directly (build_paper_paths, clean/esc
helpers) and via Streamlit's AppTest harness (api_key_configured warning).

AppTest has no file_uploader simulation support in this Streamlit version,
so file-upload-shaped cases (FU-10, FU-11) are tested against
`build_paper_paths` directly — the exact function app.py calls, extracted
verbatim for testability, not reimplemented.
"""
import os

import pytest

from app import build_paper_paths, api_key_configured, clean, esc


class FakeUpload:
    """Minimal stand-in for Streamlit's UploadedFile: .name + .getvalue()."""

    def __init__(self, name: str, content: bytes = b"dummy pdf bytes"):
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


# ── FU-10: duplicate year across two uploaded filenames ─────────────────


def test_fu10_duplicate_year_does_not_silently_drop_a_paper(tmp_path):
    """FU-10: two files whose filenames both contain the same year collide
    on the same dict key. This is expected to FAIL today — build_paper_paths
    currently returns only 1 entry for 2 uploaded files, silently discarding
    one paper's path with no warning."""
    papers = [FakeUpload("2022_sem1.pdf"), FakeUpload("2022_sem2.pdf")]
    result = build_paper_paths(papers, str(tmp_path))
    assert len(result) == len(papers), (
        f"expected one path per uploaded file, got {len(result)} for {len(papers)} "
        f"uploads — a duplicate year silently overwrote a prior entry: {result}"
    )


def test_fu10_both_files_are_at_least_written_to_disk(tmp_path):
    """Narrower sibling check: even though the dict loses a file (above),
    confirm both files are physically written under the tmpdir (so the data
    isn't technically un-recoverable, just unreferenced) — locks in current
    partial behavior."""
    papers = [FakeUpload("2022_sem1.pdf"), FakeUpload("2022_sem2.pdf")]
    build_paper_paths(papers, str(tmp_path))
    assert (tmp_path / "2022_sem1.pdf").exists()
    assert (tmp_path / "2022_sem2.pdf").exists()


# ── FU-11: path traversal via crafted filename ──────────────────────────


def test_fu11_path_traversal_filename_stays_inside_tmpdir(tmp_path):
    """FU-11 / SEC-1: a crafted filename with `../` must not let the write
    escape the sandboxed tmpdir. Expected to FAIL today — build_paper_paths
    uses a bare os.path.join(tmpdir, pf.name) with no sanitization, and a
    single '../name.pdf' filename provably writes into tmpdir's parent."""
    evil = FakeUpload("../escaped_via_traversal.pdf")
    result = build_paper_paths([evil], str(tmp_path))
    written_path = os.path.abspath(list(result.values())[0])
    tmpdir_abs = os.path.abspath(str(tmp_path))

    # cleanup regardless of outcome, so a failing test doesn't litter /tmp
    parent_escape = os.path.join(os.path.dirname(tmpdir_abs), "escaped_via_traversal.pdf")
    try:
        assert written_path.startswith(tmpdir_abs + os.sep), (
            f"path traversal escaped the sandbox: wrote to {written_path}, "
            f"which is outside {tmpdir_abs}"
        )
    finally:
        if os.path.exists(parent_escape):
            os.unlink(parent_escape)


def test_fu11_absolute_path_filename_stays_inside_tmpdir(tmp_path):
    """A filename that is itself an absolute path is an even more direct
    version of the same vulnerability class — os.path.join discards the
    first argument entirely when the second is absolute."""
    evil = FakeUpload("/tmp/absolute_path_traversal_test.pdf")
    result = build_paper_paths([evil], str(tmp_path))
    written_path = os.path.abspath(list(result.values())[0])
    tmpdir_abs = os.path.abspath(str(tmp_path))
    try:
        assert written_path.startswith(tmpdir_abs + os.sep), (
            f"absolute filename escaped the sandbox entirely: wrote to {written_path}"
        )
    finally:
        if os.path.exists("/tmp/absolute_path_traversal_test.pdf"):
            os.unlink("/tmp/absolute_path_traversal_test.pdf")


# ── SEC-2: XSS escaping regression (currently passing — lock it in) ─────


@pytest.mark.parametrize(
    "malicious_input",
    [
        "<script>alert('xss')</script>",
        "<img src=x onerror=alert(1)>",
        '"><svg onload=alert(1)>',
        "normal text with <b>bold</b> tags",
    ],
)
def test_sec2_clean_escapes_html_from_llm_output(malicious_input):
    """SEC-2: every dynamic field rendered via unsafe_allow_html in
    app.py's results panels is passed through clean()/esc() first. This
    locks that protection in — it should currently PASS."""
    out = clean(malicious_input)
    assert "<script" not in out.lower()
    assert "<img" not in out.lower()
    assert "<svg" not in out.lower()
    assert "&lt;" in out or "<" not in malicious_input.replace("<b>", "")


def test_sec2_esc_escapes_html():
    out = esc("<script>alert(1)</script>")
    assert out == "&lt;script&gt;alert(1)&lt;/script&gt;"


# ── API key warning (feature you asked for) ──────────────────────────────


def test_api_key_configured_true_when_env_set(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-fake-key-value")
    assert api_key_configured() is True


def test_api_key_configured_false_when_env_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert api_key_configured() is False


def test_api_key_configured_false_when_env_blank(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    assert api_key_configured() is False


def test_api_key_warning_shown_on_page_load_when_missing(monkeypatch):
    """Drives the real Streamlit script via AppTest with no API key set —
    confirms the warning banner from render_input() actually appears.

    AppTest re-executes app.py's source fresh, which re-runs load_dotenv()
    and would silently restore GEMINI_API_KEY from the real .env file —
    so dotenv's loader itself must be neutralized, not just os.environ."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: False)
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py"), default_timeout=30)
    at.run()
    assert at.exception == []
    warning_texts = " ".join(e.value for e in at.error)
    assert "GEMINI_API_KEY" in warning_texts


def test_no_api_key_warning_when_configured(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "sk-fake-key-value")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py"), default_timeout=30)
    at.run()
    assert at.exception == []
    warning_texts = " ".join(e.value for e in at.error)
    assert "GEMINI_API_KEY" not in warning_texts
