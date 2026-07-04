import streamlit as st
import tempfile
import os
import re
import time
import html
import threading
from dotenv import load_dotenv
from pipeline.graph import run_pipeline
from utils.pdf_generator import generate_practice_paper_pdf, _clean_latex

load_dotenv()


def api_key_configured() -> bool:
    """True if GEMINI_API_KEY is set to a non-empty value in the environment."""
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


MAX_PAPERS = 6  # Streamlit's file_uploader has no built-in max-file-count option
MAX_SLIDES = 6  # so this is enforced explicitly at submit time (see render_input)


def _safe_upload_name(name: str) -> str:
    """Reduce a client-supplied upload filename to a safe basename.

    An uploaded file's ``.name`` is fully attacker-controlled, so writing it
    with a bare ``os.path.join(tmpdir, name)`` lets ``../`` or an absolute path
    escape the temp dir. Stripping to the basename (and rejecting empty/dot
    names) keeps every write confined to ``tmpdir``.
    """
    base = os.path.basename(name or "")
    # Defend against Windows-style separators the POSIX basename won't split,
    # plus empty / dot-only names.
    base = base.replace("\\", "/").rsplit("/", 1)[-1]
    if not base or base in {".", ".."}:
        base = "upload.pdf"
    return base


def build_paper_paths(papers: list, tmpdir: str) -> dict:
    """Write each uploaded paper to disk and map it to a detected year.

    Extracted as a standalone function (unchanged behavior) so it's
    directly unit-testable without driving the full Streamlit UI — Streamlit's
    AppTest harness has no file_uploader simulation support.

    ``papers`` items only need ``.name`` (str) and ``.getvalue()`` (bytes),
    matching Streamlit's UploadedFile duck-type.
    """
    paper_paths = {}
    for i, pf in enumerate(papers):
        path = os.path.join(tmpdir, _safe_upload_name(pf.name))
        with open(path, "wb") as f:
            f.write(pf.getvalue())
        year_match = re.search(r"(20\d\d)", pf.name)
        year = year_match.group(1) if year_match else str(2020 + i)
        paper_paths[year] = path
    return paper_paths


def build_slide_paths(files: list, tmpdir: str) -> list:
    """Write each uploaded slide deck to disk and return their paths.

    Mirrors ``build_paper_paths`` — extracted the same way for testability.
    """
    paths = []
    for sf in files:
        path = os.path.join(tmpdir, _safe_upload_name(sf.name))
        with open(path, "wb") as f:
            f.write(sf.getvalue())
        paths.append(path)
    return paths


st.set_page_config(
    page_title="ExamCast — Predict my exam",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="collapsed",
)

esc = html.escape


def clean(text: str) -> str:
    """Escape for safe HTML embedding, after light LaTeX/math cleanup."""
    return esc(_clean_latex(text or ""))


# ── Session state ────────────────────────────────────────────

DEFAULTS = {
    "screen": "input",
    "subject": "",
    "tab": "topics",
    "tmpdir": None,
    "paper_paths": {},
    "slides_path": None,
    "has_slides": False,
    "pipeline_box": None,
    "gen_start": None,
    "result": None,
    "pdf_bytes": None,
    "flash_error": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

MSGS = [
    "Cramming? So are we — hang tight.",
    "Spotting the questions your lecturer loves.",
    "Learning how your exam likes to phrase things.",
    "Writing questions in your paper's style.",
    "Almost there — building your answer key.",
]
STAGE_DEFS = [
    ("Ingest", "reading PDFs"),
    ("Analyze", "finding hot topics"),
    ("Style profile", "matching format"),
    ("Generate", "writing paper"),
]
SIM_DURATION = 45.0  # seconds — calibrated to average real pipeline runtime


# ── CSS ──────────────────────────────────────────────────────


def inject_base_css():
    st.markdown(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Newsreader:ital,opsz@1,6..72&display=swap" rel="stylesheet">',
        unsafe_allow_html=True,
    )
    st.markdown(
        """<style>
html, body, [class^="st-"], [class*=" st-"] { font-family:'Space Grotesk',system-ui,sans-serif; }
.stApp { background:#f3efe6; }
header[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer { visibility:hidden; }
.serif { font-family:'Newsreader',Georgia,serif; font-style:italic; }

/* ── card shell ── */
.block-container {
  border:1px solid rgba(0,0,0,.08);
  border-radius:22px;
  background:#fffdf7;
  box-shadow:0 1px 2px rgba(0,0,0,.04),0 24px 48px -28px rgba(60,40,20,.4);
  margin-top:48px; margin-bottom:80px;
}

/* ── brand / headings ── */
.brand{display:flex;align-items:center;gap:9px;font-weight:700;font-size:15px;letter-spacing:-.01em;margin-bottom:26px}
.mark{width:22px;height:22px;border-radius:50%;background:conic-gradient(from 140deg,#d1622b 0 50%,#26231d 50% 100%);flex:none;display:inline-block}
.bpill{font-size:10px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#a84a1c;background:#f4e2d3;padding:3px 8px;border-radius:20px;margin-left:auto}
.disp{font-size:38px;line-height:1.02;letter-spacing:-.03em;font-weight:700;margin:0 0 12px;color:#26231d}
.lead{font-size:15px;line-height:1.5;color:#6f695c;margin:0 0 22px;max-width:44ch}
.fldlab{display:flex;align-items:baseline;gap:8px;font-size:13px;font-weight:600;margin:0 0 8px;color:#26231d}
.hint{font-weight:400;color:#9b9382;font-size:12px}
.reassure{text-align:center;font-size:13px;color:#9b9382;margin:14px 0 0}

/* ── field / uploader skinning ── */
div[data-testid="stTextInput"] input{
  height:48px;border:1.5px solid rgba(0,0,0,.14)!important;border-radius:12px!important;
  background:#fff;padding:0 15px;font:500 15px 'Space Grotesk';color:#26231d;
}
div[data-testid="stTextInput"] input:focus{
  border-color:#d1622b!important;box-shadow:0 0 0 4px rgba(209,98,43,.12)!important;
}
div[data-testid="stTextInput"]{margin-bottom:22px}

[data-testid="stFileUploaderDropzone"]{
  border:1.5px dashed rgba(0,0,0,.2)!important;border-radius:14px!important;background:#faf6ee!important;
}
[data-testid="stFileUploaderDropzone"]:hover{border-color:#d1622b!important;background:#fcf2e9!important}
[data-testid="stFileUploaderDropzone"] button{border:1.5px solid rgba(209,98,43,.5)!important;color:#a84a1c!important;border-radius:9px!important;background:#fff!important}
[data-testid="stFileUploaderFile"]{background:#fff;border:1px solid rgba(0,0,0,.1);border-radius:10px}
div[data-testid="stFileUploader"]{margin-bottom:22px}

/* ── alerts ── */
[data-testid="stAlert"]{
  border-radius:12px;background:#fbeee6!important;border:1px solid rgba(209,98,43,.25)!important;
  color:#a84a1c!important;margin-bottom:22px;
}
[data-testid="stAlert"] p{color:#a84a1c!important;font-weight:500}

/* ── buttons: base / default = accent filled pill (CTA) ── */
.stButton button[kind="primary"], .stDownloadButton button[kind="primary"]{
  width:100%;height:54px;border:none;border-radius:14px;background:#d1622b;color:#fff!important;
  font-weight:700;font-size:16px;letter-spacing:-.01em;
  box-shadow:0 12px 24px -12px rgba(209,98,43,.8);transition:transform .12s,background .15s;
}
.stButton button[kind="primary"]:hover, .stDownloadButton button[kind="primary"]:hover{background:#c05622;transform:translateY(-1px)}
.stButton button[kind="secondary"], .stDownloadButton button[kind="secondary"]{
  width:100%;height:48px;border:1.5px solid rgba(0,0,0,.14);background:#fff;color:#26231d;
  border-radius:11px;font-weight:600;font-size:14px;
}
.stButton button[kind="secondary"]:hover, .stDownloadButton button[kind="secondary"]:hover{background:#faf6ee}

/* ── generating screen ── */
.gcard-inner{text-align:center;padding:10px 4px 4px}
.loader{width:76px;height:76px;margin:0 auto 24px;position:relative}
.loader .ring{position:absolute;inset:0;border-radius:50%;border:4px solid #f1e3d5;border-top-color:#d1622b;animation:sp 1.1s cubic-bezier(.5,.15,.5,.85) infinite}
.loader .ring2{inset:12px;border-width:3px;border-top-color:#26231d;animation-duration:1.7s;animation-direction:reverse}
@keyframes sp{to{transform:rotate(360deg)}}
.gh{font-size:24px;font-weight:700;letter-spacing:-.02em;margin:0 0 8px;color:#26231d}
.quote{font-size:17px;color:#8a8272;margin:0 auto 26px;max-width:36ch;min-height:1.4em;text-align:center}
.track{height:8px;border-radius:8px;background:#eee4d6;overflow:hidden;margin:0 0 8px}
.track>i{display:block;height:100%;background:linear-gradient(90deg,#d1622b,#e07d43);border-radius:8px;transition:width .5s linear}
.pctrow{display:flex;justify-content:space-between;font-size:12px;font-weight:600;color:#9b9382;margin-bottom:26px}
.stages{list-style:none;margin:0;padding:0;text-align:left;display:flex;flex-direction:column;gap:2px}
.stg{display:flex;align-items:center;gap:13px;padding:11px 12px;border-radius:12px;transition:background .2s}
.stg .sd{width:22px;height:22px;border-radius:50%;border:2px solid rgba(0,0,0,.16);flex:none;display:flex;align-items:center;justify-content:center;font-size:12px;color:#fff;transition:.2s}
.stg .sl{font-weight:600;font-size:14px;color:#a49c8c}
.stg .ssub{font-size:12px;color:#c0b8a6;margin-left:auto}
.stg.active{background:#fbf1e8}
.stg.active .sd{border-color:#d1622b}
.stg.active .sl{color:#26231d}
.stg.active .ssub{color:#d1622b}
.stg.done .sd{background:#26231d;border-color:#26231d}
.stg.done .sl{color:#6f695c}
.stg.done .ssub{color:transparent}

/* ── results: header / topics / style / paper ── */
.rhead{display:flex;align-items:flex-start;justify-content:space-between;gap:20px;padding:2px 0 22px;border-bottom:1px solid rgba(0,0,0,.07);margin-bottom:4px}
.rkicker{display:inline-flex;align-items:center;gap:7px;font-size:12px;font-weight:600;color:#4f8a5b;margin-bottom:8px}
.rkicker .tick{width:16px;height:16px;border-radius:50%;background:#4f8a5b;color:#fff;display:inline-flex;align-items:center;justify-content:center;font-size:10px}
.rtitle{font-size:26px;font-weight:700;letter-spacing:-.02em;margin:0;color:#26231d}
.rsub{font-size:13px;color:#9b9382;margin:5px 0 0}

.topic{display:flex;gap:18px;padding:20px 0;border-bottom:1px solid rgba(0,0,0,.06)}
.topic:first-child{padding-top:6px}
.topic:last-child{border-bottom:none;padding-bottom:2px}
.trank{font-family:'Newsreader',serif;font-style:italic;font-size:30px;color:#d1622b;line-height:1;flex:none;width:44px}
.tbody{flex:1;min-width:0}
.ttop{display:flex;align-items:baseline;justify-content:space-between;gap:14px;flex-wrap:wrap;margin-bottom:12px}
.tname{font-size:18px;font-weight:700;letter-spacing:-.01em;margin:0;color:#26231d}
.tmeta{display:flex;align-items:center;gap:10px;flex:none}
.freq{font-size:12px;font-weight:700;color:#a84a1c;background:#f4e2d3;padding:4px 10px;border-radius:20px}
.yrs{font-size:12px;color:#9b9382}
.concept{padding:9px 0 9px 16px;border-left:2px solid #eaddcc;margin-bottom:4px}
.cname{font-weight:600;font-size:14px;color:#26231d}
.cdesc{color:#6f695c;font-size:14px}
.ceg{display:block;font-size:12.5px;color:#a49c8c;font-style:italic;margin-top:3px}

.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:24px}
.metric{background:#faf6ee;border:1px solid rgba(0,0,0,.06);border-radius:14px;padding:18px}
.mval{font-family:'Newsreader',serif;font-style:italic;font-size:34px;line-height:1;color:#26231d}
.mlab{font-size:12px;font-weight:600;color:#9b9382;margin-top:8px}
.drow{display:flex;gap:14px;padding:14px 0;border-bottom:1px solid rgba(0,0,0,.06)}
.drow:last-child{border-bottom:none}
.dk{font-size:13px;font-weight:600;color:#9b9382;width:150px;flex:none}
.dv{font-size:14px;color:#3a352c;flex:1}
.verbs{display:flex;flex-wrap:wrap;gap:7px}
.verb{font-size:13px;font-weight:600;background:#f4e2d3;color:#a84a1c;padding:4px 11px;border-radius:8px}

.psec{margin-bottom:24px}
.psec:last-child{margin-bottom:0}
.psech{display:flex;align-items:baseline;justify-content:space-between;border-bottom:2px solid #26231d;padding-bottom:7px;margin-bottom:14px}
.psect{font-size:15px;font-weight:700;letter-spacing:.02em;text-transform:uppercase;color:#26231d}
.psecm{font-size:13px;font-weight:600;color:#9b9382}
.q{display:flex;gap:14px;padding:12px 0}
.qn{font-weight:700;font-size:15px;flex:none;width:26px;color:#26231d}
.qt{flex:1;font-size:14.5px;line-height:1.55;color:#3a352c}
.qmk{font-weight:700;font-size:13px;color:#a84a1c;flex:none}
.qsub{display:block;color:#6f695c;margin-top:6px;padding-left:2px}
.pnote{margin-top:8px;font-size:13px;color:#9b9382;background:#faf6ee;border-radius:12px;padding:14px 16px;text-align:center}
</style>
""",
        unsafe_allow_html=True,
    )


def inject_card_css(wide: bool):
    if wide:
        st.markdown(
            "<style>.block-container{max-width:900px!important;padding:0 38px 30px!important}</style>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<style>.block-container{max-width:620px!important;padding:38px 40px 34px!important}</style>",
            unsafe_allow_html=True,
        )


def inject_scoped_css():
    """Marker-scoped overrides for button groups that must look different
    from the default accent-pill button (tabs, downloads, start-over)."""
    st.markdown(
        """
<style>
/* tab bar: flat underline tabs, not filled pills */
div.element-container:has(#examcast-tab-marker) + div[data-testid="stHorizontalBlock"] button{
  width:auto!important;height:auto!important;padding:13px 4px!important;border:none!important;
  border-bottom:2.5px solid transparent!important;border-radius:0!important;background:none!important;
  box-shadow:none!important;font:600 14px 'Space Grotesk'!important;color:#9b9382!important;
}
div.element-container:has(#examcast-tab-marker) + div[data-testid="stHorizontalBlock"] button[kind="primary"]{
  color:#26231d!important;border-bottom-color:#d1622b!important;
}
div.element-container:has(#examcast-tab-marker) + div[data-testid="stHorizontalBlock"]{
  border-bottom:1px solid rgba(0,0,0,.07);padding-bottom:0;margin-bottom:10px;
}

/* download buttons: small pills, not full-height CTA */
div.element-container:has(#examcast-dl-marker) + div[data-testid="stHorizontalBlock"] button{
  height:42px!important;border-radius:11px!important;font-size:14px!important;box-shadow:none!important;
}

/* start-over: plain centered text link */
div.element-container:has(#examcast-reset-marker) + div[data-testid="stHorizontalBlock"] button,
div.element-container:has(#examcast-reset-marker) ~ div.element-container button[kind="secondary"]{
  width:auto!important;height:auto!important;border:none!important;background:none!important;box-shadow:none!important;
  color:#9b9382!important;font:600 13px 'Space Grotesk'!important;padding:10px 16px!important;display:block;margin:0 auto;
}
</style>
""",
        unsafe_allow_html=True,
    )


# ── Pipeline kickoff ─────────────────────────────────────────


def start_pipeline(subject: str, slides_path, paper_paths: dict):
    box = {"done": False, "result": None, "error": None}

    def _worker():
        try:
            box["result"] = run_pipeline(subject, slides_path, paper_paths)
        except Exception as e:
            box["error"] = str(e)
        finally:
            box["done"] = True

    t = threading.Thread(target=_worker, daemon=True)
    st.session_state.pipeline_box = box
    st.session_state.gen_start = time.time()
    st.session_state.screen = "generating"
    t.start()


# ── Screens ──────────────────────────────────────────────────


def render_input():
    inject_card_css(wide=False)

    if not api_key_configured():
        st.error(
            "⚠️ GEMINI_API_KEY is not set. Generation will fail until it's "
            "configured (add it to your .env file, or your deployment's secrets)."
        )

    if st.session_state.flash_error:
        st.error(st.session_state.flash_error)
        st.session_state.flash_error = None

    st.markdown(
        """
<div class="brand"><span class="mark"></span>ExamCast<span class="bpill">beta</span></div>
<div class="disp">Predict my exam.</div>
<p class="lead">Upload your past papers — we'll find the patterns and write you a full practice paper in your exam's own style.</p>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="fldlab">Subject</div>', unsafe_allow_html=True)
    subject_val = st.text_input(
        "Subject",
        value=st.session_state.subject,
        placeholder="e.g. DSA1101 Introduction to Data Science",
        key="subject_input",
        label_visibility="collapsed",
    )

    st.markdown(
        f'<div class="fldlab">Past year papers <span class="hint">3–5 PDFs, one per year — up to {MAX_PAPERS} files</span></div>',
        unsafe_allow_html=True,
    )
    papers = st.file_uploader(
        "Past year papers",
        type=["pdf"],
        accept_multiple_files=True,
        key="papers_uploader",
        label_visibility="collapsed",
    )

    st.markdown(
        f'<div class="fldlab">Lecture slides <span class="hint">optional — sharpens topic predictions — up to {MAX_SLIDES} files</span></div>',
        unsafe_allow_html=True,
    )
    slides = st.file_uploader(
        "Lecture slides",
        type=["pdf", "pptx"],
        accept_multiple_files=True,
        key="slides_uploader",
        label_visibility="collapsed",
    )

    generate_clicked = st.button(
        "Generate practice paper", type="primary", use_container_width=True
    )
    st.markdown(
        '<p class="reassure">Takes about 45 seconds. No sign-up.</p>',
        unsafe_allow_html=True,
    )

    if generate_clicked:
        subject_clean = (subject_val or "").strip()
        papers = papers or []
        slides = slides or []
        if not subject_clean:
            st.session_state.flash_error = "Please enter a subject name."
            st.rerun()
        elif not papers:
            st.session_state.flash_error = "Please upload at least one past year paper."
            st.rerun()
        elif len(papers) > MAX_PAPERS:
            st.session_state.flash_error = (
                f"Please upload at most {MAX_PAPERS} past year papers (you uploaded {len(papers)})."
            )
            st.rerun()
        elif len(slides) > MAX_SLIDES:
            st.session_state.flash_error = (
                f"Please upload at most {MAX_SLIDES} lecture slide files (you uploaded {len(slides)})."
            )
            st.rerun()
        elif not api_key_configured():
            st.session_state.flash_error = (
                "GEMINI_API_KEY is not configured — generation is disabled until it's set."
            )
            st.rerun()
        else:
            tmpdir = tempfile.mkdtemp(prefix="examcast_")
            paper_paths = build_paper_paths(papers, tmpdir)
            slides_path = build_slide_paths(slides, tmpdir) if slides else []

            st.session_state.subject = subject_clean
            st.session_state.tmpdir = tmpdir
            st.session_state.paper_paths = paper_paths
            st.session_state.slides_path = slides_path
            st.session_state.has_slides = bool(slides_path)

            start_pipeline(subject_clean, slides_path, paper_paths)
            st.rerun()


def render_generating():
    inject_card_css(wide=False)

    # Poll the background pipeline from a fragment so the MAIN script run
    # RUNS TO COMPLETION — that's what lets Streamlit prune the input screen's
    # widgets. The old time.sleep()+st.rerun() loop never let a run complete,
    # so on Streamlit 1.5x the input widgets were never cleared and showed
    # through under this screen. The fragment re-runs only itself every 0.7s.
    @st.fragment(run_every=0.7)
    def _poll():
        box = st.session_state.pipeline_box
        elapsed = time.time() - (st.session_state.gen_start or time.time())
        done = bool(box and box["done"])

        progress = 100.0 if done else min(95.0, (elapsed / SIM_DURATION) * 100.0)
        stage_index = min(3, int(progress // 25))
        msg_index = int(elapsed // 2.2) % len(MSGS)

        stages_html = ""
        for i, (label, sub) in enumerate(STAGE_DEFS):
            if done or i < stage_index:
                cls, subtext = "stg done", "done"
            elif i == stage_index:
                cls, subtext = "stg active", sub
            else:
                cls, subtext = "stg", sub
            stages_html += (
                f'<li class="{cls}"><span class="sd">✓</span>'
                f'<span class="sl">{label}</span><span class="ssub">{subtext}</span></li>'
            )

        st.markdown(
            f"""
<div class="gcard-inner">
  <div class="loader"><div class="ring"></div><div class="ring ring2"></div></div>
  <div class="gh">Reading your papers…</div>
  <p class="quote serif">{esc(MSGS[msg_index])}</p>
  <div class="track"><i style="width:{progress:.0f}%"></i></div>
  <div class="pctrow"><span>{progress:.0f}%</span><span>~45s</span></div>
  <ul class="stages">{stages_html}</ul>
</div>
""",
            unsafe_allow_html=True,
        )

        if not done:
            return  # fragment auto-re-runs in 0.7s to keep polling

        result = box.get("result")
        err = box.get("error") or (result.get("error") if result else "Unknown pipeline error")
        if err:
            st.session_state.screen = "input"
            st.session_state.flash_error = f"Pipeline error: {err}"
            st.session_state.pipeline_box = None
            st.rerun(scope="app")
            return

        st.session_state.result = result
        generated_json = result.get("generated_paper_json")
        if generated_json:
            try:
                st.session_state.pdf_bytes = generate_practice_paper_pdf(
                    st.session_state.subject,
                    generated_json,
                    result.get("style") or {},
                    result.get("has_slides", True),
                )
            except Exception:
                st.session_state.pdf_bytes = None
        else:
            st.session_state.pdf_bytes = None

        st.session_state.pipeline_box = None
        st.session_state.screen = "results"
        st.session_state.tab = "topics"
        st.rerun(scope="app")

    _poll()


def topic_title(topic: dict, idx: int) -> str:
    if topic.get("topic_name"):
        return topic["topic_name"]
    preview = (topic.get("preview") or topic.get("content") or "").strip()
    title = re.sub(r"\s+", " ", preview.split("\n")[0])
    if len(title) > 60:
        title = title[:60].rsplit(" ", 1)[0] + "…"
    if not title:
        concepts = topic.get("concepts") or []
        title = concepts[0]["name"] if concepts else f"Topic {idx + 1}"
    return title


def render_topics_panel(high_prob: list):
    if not high_prob:
        st.markdown(
            '<div class="panel"><p style="color:#9b9382">No topic data available.</p></div>',
            unsafe_allow_html=True,
        )
        return

    rows = []
    for i, topic in enumerate(high_prob):
        count = topic.get("count", 0)
        years = topic.get("years", [])
        year_str = ", ".join(years) if years else "—"
        concepts = topic.get("concepts") or []

        concept_html = "".join(
            f'<div class="concept"><span class="cname">{clean(c.get("name",""))}</span> '
            f'<span class="cdesc">— {clean(c.get("description",""))}</span>'
            f'<span class="ceg">e.g. {clean(c.get("sample_question_text",""))}</span></div>'
            for c in concepts
        ) or '<p style="color:#c0b8a6;font-size:13px;font-style:italic;margin:2px 0 0">Concept analysis unavailable.</p>'

        rows.append(
            f"""<div class="topic"><div class="trank">{i + 1:02d}</div><div class="tbody">
<div class="ttop"><div class="tname">{clean(topic_title(topic, i))}</div>
<div class="tmeta"><span class="freq">Tested {count}×</span><span class="yrs">{esc(year_str)}</span></div></div>
{concept_html}
</div></div>"""
        )

    st.markdown(f'<div class="panel">{"".join(rows)}</div>', unsafe_allow_html=True)


def render_style_panel(style: dict):
    verbs = style.get("common_verbs") or []
    verbs_html = "".join(f'<span class="verb">{clean(v)}</span>' for v in verbs) or "—"

    st.markdown(
        f"""
<div class="panel">
<div class="metrics">
  <div class="metric"><div class="mval">{esc(str(style.get("estimated_total_marks", "N/A")))}</div><div class="mlab">Total marks</div></div>
  <div class="metric"><div class="mval">{esc(str(style.get("total_sections", "N/A")))}</div><div class="mlab">Sections</div></div>
  <div class="metric"><div class="mval">{esc(str(style.get("marks_per_question", "N/A")))}</div><div class="mlab">Marks / question</div></div>
</div>
<div class="drow"><div class="dk">Common command verbs</div><div class="dv"><div class="verbs">{verbs_html}</div></div></div>
<div class="drow"><div class="dk">Question format</div><div class="dv">{clean(style.get("question_format","N/A"))}</div></div>
<div class="drow"><div class="dk">Sub-part pattern</div><div class="dv">{clean(style.get("subpart_pattern","N/A"))}</div></div>
<div class="drow"><div class="dk">Phrasing style</div><div class="dv">{clean(style.get("sample_question_style","N/A"))}</div></div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_paper_panel(questions: list, style: dict):
    if not questions:
        st.markdown(
            '<div class="panel"><p style="color:#9b9382">Paper generation failed.</p></div>',
            unsafe_allow_html=True,
        )
        return

    total_marks = style.get("estimated_total_marks") or sum(
        q.get("marks", 0) or 0 for q in questions
    )

    q_html = []
    for q in questions:
        subparts = q.get("subparts") or []
        sub_html = ""
        if subparts:
            parts = []
            for sp in subparts:
                sp_marks = sp.get("marks")
                mark_str = f" [{sp_marks}]" if sp_marks else ""
                parts.append(f'({clean(sp.get("label",""))}) {clean(sp.get("text",""))}{mark_str}')
            sub_html = f'<span class="qsub">{" &nbsp;&nbsp;".join(parts)}</span>'

        marks = q.get("marks")
        mark_display = f"[{marks}]" if marks else ""
        q_html.append(
            f"""<div class="q"><span class="qn">{esc(str(q.get("question_number","?")))}</span>
<span class="qt">{clean(q.get("question_text",""))}{sub_html}</span>
<span class="qmk">{esc(mark_display)}</span></div>"""
        )

    st.markdown(
        f"""
<div class="panel">
<div class="psec"><div class="psech"><span class="psect">Questions</span><span class="psecm">{esc(str(total_marks))} marks</span></div>
{"".join(q_html)}
</div>
<div class="pnote">Full worked answer key is included at the back of the PDF.</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_results():
    inject_card_css(wide=True)
    inject_scoped_css()

    result = st.session_state.result
    subject = result.get("subject", st.session_state.subject)
    high_prob = result.get("high_prob_topics") or []
    style = result.get("style") or {}
    generated_json = result.get("generated_paper_json") or []
    generated_text = result.get("generated_paper") or ""

    n_papers = len(st.session_state.paper_paths)
    n_questions = len(generated_json)
    total_marks = style.get("estimated_total_marks") or sum(
        q.get("marks", 0) or 0 for q in generated_json
    )

    st.markdown(
        f"""
<div class="rhead">
  <div>
    <div class="rkicker"><span class="tick">✓</span>Practice paper ready</div>
    <div class="rtitle">{clean(subject)}</div>
    <p class="rsub">Predicted from {n_papers} past paper{"s" if n_papers != 1 else ""} · {n_questions} questions · {esc(str(total_marks))} marks</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div id="examcast-dl-marker"></div>', unsafe_allow_html=True)
    spacer, c1, c2 = st.columns([5, 2, 2])
    with c1:
        pdf_bytes = st.session_state.pdf_bytes
        if pdf_bytes:
            st.download_button(
                "⬇ PDF",
                data=pdf_bytes,
                file_name=f"{subject.replace(' ', '_')}_practice_paper.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
                key="dl_pdf",
            )
        else:
            st.button("⬇ PDF", disabled=True, use_container_width=True, key="dl_pdf_disabled")
    with c2:
        st.download_button(
            "TXT",
            data=generated_text,
            file_name=f"{subject.replace(' ', '_')}_practice_paper.txt",
            mime="text/plain",
            use_container_width=True,
            key="dl_txt",
        )

    st.markdown('<div id="examcast-tab-marker"></div>', unsafe_allow_html=True)
    tabs = [
        ("topics", f"Topic analysis  {len(high_prob)}"),
        ("style", "Style profile"),
        ("paper", "Practice paper"),
    ]
    tab_cols = st.columns(3)
    for col, (key, label) in zip(tab_cols, tabs):
        with col:
            is_on = st.session_state.tab == key
            if st.button(
                label,
                key=f"tabbtn_{key}",
                type="primary" if is_on else "secondary",
                use_container_width=True,
            ):
                st.session_state.tab = key
                st.rerun()

    if st.session_state.tab == "topics":
        render_topics_panel(high_prob)
    elif st.session_state.tab == "style":
        render_style_panel(style)
    else:
        render_paper_panel(generated_json, style)

    st.markdown('<div id="examcast-reset-marker"></div>', unsafe_allow_html=True)
    if st.button("↺ Start over", key="reset_btn"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()


# ── Router ───────────────────────────────────────────────────

inject_base_css()

# Render the active screen into a single placeholder so switching screens
# REPLACES the previous screen's content instead of stacking on top of it.
# The "generating" screen polls via time.sleep()+st.rerun() and so never runs
# to completion; since Streamlit 1.4x defers cleanup of a prior run's widgets
# until a run completes, the input screen's widgets would otherwise stay
# painted underneath it. A shared st.empty() slot forces the replacement.
if st.session_state.screen == "input":
    render_input()
elif st.session_state.screen == "generating":
    render_generating()
elif st.session_state.screen == "results":
    render_results()
