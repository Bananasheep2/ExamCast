import threading

from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

from pipeline.ingest import ingest_slides, ingest_past_paper
from pipeline.analyzer import (
    enrich_topics_with_concepts,
    build_topic_frequency_map_no_slides,
    get_high_probability_topics_no_slides,
    get_untested_topics_no_slides,
)
from pipeline.style_extractor import extract_paper_style
from pipeline.generator import generate_practice_paper


class PipelineState(TypedDict):
    subject: str
    slides_paths: list  # empty when no slides uploaded
    has_slides: bool  # lecture notes present — used only to ground generation
    paper_paths: dict
    slides_chunks: int
    paper_chunks: int
    frequency_map: Optional[dict]
    high_prob_topics: Optional[list]
    untested_topics: Optional[list]
    style: Optional[dict]
    generated_paper: Optional[str]  # raw text (backward compat)
    generated_paper_json: Optional[list]  # structured questions for PDF
    error: Optional[str]


# ── Pipeline nodes ─────────────────────────────────────────


def _ingest_papers_concurrently(paper_paths: dict, subject: str) -> int:
    """Ingest every paper of one run in parallel and return the total chunk
    count. Running them concurrently makes them a single ingest batch (see
    pipeline.ingest), so the run clears stale data once and keeps every paper.
    Any worker error is re-raised so node_ingest reports it as a pipeline error.
    """
    items = list(paper_paths.items())
    if not items:
        return 0

    results: dict[str, int] = {}
    errors: list[Exception] = []
    lock = threading.Lock()

    def worker(key: str, path: str):
        # build_paper_paths disambiguates same-year uploads as "2022#2"; the
        # real year is the part before "#", and the full key namespaces the
        # paper's chunk ids so two same-year papers don't collide.
        year = key.split("#", 1)[0]
        try:
            n = ingest_past_paper(path, subject, year, paper_id=key)
            with lock:
                results[key] = n
        except Exception as e:  # noqa: BLE001 — surfaced below via node_ingest
            with lock:
                errors.append(e)

    threads = [threading.Thread(target=worker, args=(k, p)) for k, p in items]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        raise errors[0]
    return sum(results.values())


def node_ingest(state: PipelineState) -> PipelineState:
    """Ingest slides (if available) and past papers into ChromaDB."""
    try:
        n_slides = 0
        if state["has_slides"] and state["slides_paths"]:
            n_slides = ingest_slides(state["slides_paths"], state["subject"])

        # Ingest all of this run's papers concurrently so they register as a
        # single ingest batch: the batch clears the prior run's data once and
        # then accumulates every paper, instead of a sequential loop in which
        # each call would look like a new run and wipe the one before it.
        n_papers = _ingest_papers_concurrently(
            state["paper_paths"], state["subject"])

        return {**state, "slides_chunks": n_slides, "paper_chunks": n_papers}
    except Exception as e:
        return {**state, "error": f"Ingest failed: {str(e)}"}


def node_analyze_no_slides(state: PipelineState) -> PipelineState:
    """Gemini-based topic extraction from past papers only.

    Topic analysis is always papers-only — lecture notes never influence which
    topics are surfaced (they only ground generation), so this is the single
    analysis path regardless of whether notes were uploaded.
    """
    if state.get("error"):
        return state
    try:
        freq_map = build_topic_frequency_map_no_slides(state["subject"])
        high_prob = get_high_probability_topics_no_slides(freq_map, top_n=3)
        high_prob = enrich_topics_with_concepts(high_prob, state["subject"])
        untested = get_untested_topics_no_slides(freq_map)
        return {
            **state,
            "frequency_map": freq_map,
            "high_prob_topics": high_prob,
            "untested_topics": untested,
        }
    except Exception as e:
        return {**state, "error": f"Analysis (no-slides) failed: {str(e)}"}


def node_extract_style(state: PipelineState) -> PipelineState:
    """Extract exam style profile from past paper questions."""
    if state.get("error"):
        return state
    try:
        style = extract_paper_style(state["subject"])
        return {**state, "style": style}
    except Exception as e:
        return {**state, "error": f"Style extraction failed: {str(e)}"}


def node_generate(state: PipelineState) -> PipelineState:
    """Generate the practice paper using Gemini."""
    if state.get("error"):
        return state
    try:
        raw_text, structured = generate_practice_paper(
            state["subject"],
            state["high_prob_topics"],
            state["untested_topics"],
            state["style"],
            num_questions=5,
            has_slides=state["has_slides"],
        )
        return {
            **state,
            "generated_paper": raw_text,
            "generated_paper_json": structured,
        }
    except Exception as e:
        return {**state, "error": f"Generation failed: {str(e)}"}


# ── Pipeline builder ───────────────────────────────────────


def build_pipeline():
    graph = StateGraph(PipelineState)

    graph.add_node("ingest", node_ingest)
    graph.add_node("analyze_no_slides", node_analyze_no_slides)
    graph.add_node("extract_style", node_extract_style)
    graph.add_node("generate", node_generate)

    graph.set_entry_point("ingest")

    # Topic analysis is always papers-only. On an ingest error, skip straight
    # to generate so the error is surfaced instead of running analysis.
    def route_after_ingest(state: PipelineState) -> str:
        if state.get("error"):
            return "generate"
        return "analyze_no_slides"

    graph.add_conditional_edges(
        "ingest",
        route_after_ingest,
        {
            "analyze_no_slides": "analyze_no_slides",
            "generate": "generate",
        },
    )

    graph.add_edge("analyze_no_slides", "extract_style")
    graph.add_edge("extract_style", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


PIPELINE = build_pipeline()


# ── Public entry point ─────────────────────────────────────


def run_pipeline(
    subject: str,
    slides_path,
    paper_paths: dict,
) -> PipelineState:
    """Run the full exam prediction pipeline.

    Args:
        subject: Name of the subject.
        slides_path: Path(s) to lecture slides (PDF/PPTX) — a single path
            string, a list of path strings, or None/"" if unavailable.
        paper_paths: Dict mapping year strings to PDF file paths.

    Returns:
        PipelineState with all results (topic analysis, style, generated paper).
    """
    if isinstance(slides_path, str):
        slides_paths = [slides_path] if slides_path else []
    else:
        slides_paths = list(slides_path) if slides_path else []

    initial_state: PipelineState = {
        "subject": subject,
        "slides_paths": slides_paths,
        "has_slides": bool(slides_paths),
        "paper_paths": paper_paths,
        "slides_chunks": 0,
        "paper_chunks": 0,
        "frequency_map": None,
        "high_prob_topics": None,
        "untested_topics": None,
        "style": None,
        "generated_paper": None,
        "generated_paper_json": None,
        "error": None,
    }
    return PIPELINE.invoke(initial_state)
