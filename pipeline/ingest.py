import chromadb
from sentence_transformers import SentenceTransformer
from utils.pdf_utils import (
    extract_text_from_pdf, extract_text_from_pptx,
    chunk_slides, chunk_slides_by_tokens, chunk_past_paper
)
import os
import threading
from collections import defaultdict

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ── Concurrency control for ingestion ──────────────────────────────────────
# ChromaDB collections are keyed only by subject name and globally shared, with
# no per-session isolation. Two sessions ingesting the same subject concurrently
# used to race on delete/create + add and raise InvalidCollectionException /
# UniqueConstraintError, silently losing data. We serialize all mutating ops on
# a given collection with a per-collection lock (in-process; the real concern is
# concurrent uploads within one Streamlit server process).
_locks_guard = threading.Lock()
_collection_locks: dict[str, threading.RLock] = {}


def _lock_for(collection_name: str) -> threading.RLock:
    with _locks_guard:
        lock = _collection_locks.get(collection_name)
        if lock is None:
            lock = threading.RLock()
            _collection_locks[collection_name] = lock
        return lock


# Past-paper "run batching": a burst of *concurrent* ingests for one subject is
# treated as a single logical submission (a run) — the first to touch the
# collection clears the previous run's data, and the rest of the batch
# accumulate. A later ingest that arrives after the batch has fully drained
# starts a fresh run and clears again. This lets a multi-paper run keep every
# paper while a genuine re-run (uploading a different set later) does not leak
# stale years. See pipeline.graph.node_ingest, which ingests a run's papers
# concurrently so they form one batch.
_batch_guard = threading.Lock()
_batch_active: dict[str, int] = defaultdict(int)   # in-flight ingests / collection
_batch_open: dict[str, bool] = defaultdict(bool)   # is a run currently open?


def _clear_collection(collection) -> None:
    """Remove every document from a collection (without dropping the collection
    object, so no concurrent holder is left with a dangling handle)."""
    existing = collection.get(include=[])["ids"]
    if existing:
        collection.delete(ids=existing)


def _resolve_chunk_config(chunk_size, overlap, mode):
    """Merge explicit args with EXAMCAST_CHUNK_* env vars, then defaults.

    Lets the slide chunking be swept without editing code:
      EXAMCAST_CHUNK_MODE     = "page" (default) | "tokens"
      EXAMCAST_CHUNK_SIZE     = pages per chunk (page mode) or tokens (token mode)
      EXAMCAST_CHUNK_OVERLAP  = overlapping pages/tokens between neighbours
    Explicit call arguments take precedence over env vars. The defaults
    (page mode, size 3, overlap 0) reproduce the original behaviour exactly.
    """
    mode = (mode or os.getenv("EXAMCAST_CHUNK_MODE") or "page").lower()
    default_size = 3 if mode == "page" else 512
    default_overlap = 0 if mode == "page" else 64
    if chunk_size is None:
        chunk_size = int(os.getenv("EXAMCAST_CHUNK_SIZE", default_size))
    if overlap is None:
        overlap = int(os.getenv("EXAMCAST_CHUNK_OVERLAP", default_overlap))
    return mode, chunk_size, overlap


def chunk_slides_configured(pages, chunk_size=None, overlap=None, mode=None):
    """Chunk slide pages using the resolved (arg/env/default) chunk config."""
    mode, chunk_size, overlap = _resolve_chunk_config(chunk_size, overlap, mode)
    if mode == "tokens":
        return chunk_slides_by_tokens(
            pages, EMBED_MODEL.tokenizer, max_tokens=chunk_size, overlap=overlap)
    return chunk_slides(pages, chunk_size=chunk_size, overlap=overlap)

def get_chroma_client():
    # EXAMCAST_CHROMA_PATH lets tests (and the E2E suite's subprocess-launched
    # server) point at a throwaway directory instead of the real store.
    # Unset in normal operation, so default behavior is unchanged.
    path = os.getenv("EXAMCAST_CHROMA_PATH", "./chroma_db")
    return chromadb.PersistentClient(path=path)

def ingest_slides(file_paths, subject: str, chunk_size=None,
                  overlap=None, mode=None) -> int:
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    all_chunks = []
    for file_idx, file_path in enumerate(file_paths):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            pages = extract_text_from_pdf(file_path)
        elif ext in [".pptx", ".ppt"]:
            pages = extract_text_from_pptx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        chunks = chunk_slides_configured(pages, chunk_size, overlap, mode)
        for c in chunks:
            c["chunk_id"] = f"{file_idx}_{c['chunk_id']}"
        all_chunks.extend(chunks)

    if not all_chunks:
        names = ", ".join(os.path.basename(p) for p in file_paths)
        raise ValueError(
            f"No extractable text found in the uploaded slides ({names}) — "
            "they may be scanned/image-only with no text layer."
        )

    client = get_chroma_client()
    collection_name = f"slides_{subject.replace(' ', '_').lower()}"

    texts = [c["text"] for c in all_chunks]
    embeddings = EMBED_MODEL.encode(texts).tolist()
    ids = [c["chunk_id"] for c in all_chunks]
    metadatas = [{"pages": str(c["pages"]), "source": "slides"} for c in all_chunks]

    # Slides are a full replace of the subject's deck. Serialize the
    # clear-then-add so concurrent ingests for the same subject can't race on
    # a delete/create pair (which raised InvalidCollectionException before).
    with _lock_for(collection_name):
        collection = client.get_or_create_collection(collection_name)
        _clear_collection(collection)
        collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)
    return len(all_chunks)

def ingest_past_paper(file_path: str, subject: str, year: str,
                      paper_id: str | None = None) -> int:
    client = get_chroma_client()
    collection_name = f"papers_{subject.replace(' ', '_').lower()}"

    # Join the current ingest batch as early as possible (before the slow PDF
    # read/embed) so concurrent ingests reliably overlap and are recognised as
    # one run. Balanced by the decrement in `finally`.
    with _batch_guard:
        _batch_active[collection_name] += 1
    try:
        pages = extract_text_from_pdf(file_path)
        chunks = chunk_past_paper(pages, paper_year=year, paper_id=paper_id)
        if not chunks:
            # Invalid paper: raise without mutating the collection.
            raise ValueError(
                f"No extractable text found in the paper for {year} "
                f"({os.path.basename(file_path)}) — it may be a scanned/image-only "
                "PDF with no text layer, or its content couldn't be split into questions."
            )

        texts = [c["text"] for c in chunks]
        embeddings = EMBED_MODEL.encode(texts).tolist()
        ids = [c["chunk_id"] for c in chunks]
        metadatas = [{"year": c["year"], "question_num": str(c["question_num"]),
                      "source": "past_paper"} for c in chunks]

        with _lock_for(collection_name):
            # First entrant of a fresh run clears the previous run's papers
            # (no stale-year leak); later entrants in the same batch accumulate.
            with _batch_guard:
                first_in_run = not _batch_open[collection_name]
                _batch_open[collection_name] = True
            collection = client.get_or_create_collection(collection_name)
            if first_in_run:
                _clear_collection(collection)
            else:
                # Same run re-uploading the same year: replace just that year.
                try:
                    collection.delete(where={"year": year})
                except Exception:
                    pass
            collection.add(documents=texts, embeddings=embeddings, ids=ids,
                           metadatas=metadatas)
        return len(chunks)
    finally:
        with _batch_guard:
            _batch_active[collection_name] -= 1
            if _batch_active[collection_name] <= 0:
                _batch_active[collection_name] = 0
                _batch_open[collection_name] = False