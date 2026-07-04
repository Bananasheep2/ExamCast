import chromadb
from sentence_transformers import SentenceTransformer
from utils.pdf_utils import (
    extract_text_from_pdf, extract_text_from_pptx,
    chunk_slides, chunk_past_paper
)
import os

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def get_chroma_client():
    # EXAMCAST_CHROMA_PATH lets tests (and the E2E suite's subprocess-launched
    # server) point at a throwaway directory instead of the real store.
    # Unset in normal operation, so default behavior is unchanged.
    path = os.getenv("EXAMCAST_CHROMA_PATH", "./chroma_db")
    return chromadb.PersistentClient(path=path)

def ingest_slides(file_paths, subject: str) -> int:
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

        chunks = chunk_slides(pages, chunk_size=3)
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

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(collection_name)
    texts = [c["text"] for c in all_chunks]
    embeddings = EMBED_MODEL.encode(texts).tolist()
    ids = [c["chunk_id"] for c in all_chunks]
    metadatas = [{"pages": str(c["pages"]), "source": "slides"} for c in all_chunks]

    collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)
    return len(all_chunks)

def ingest_past_paper(file_path: str, subject: str, year: str) -> int:
    pages = extract_text_from_pdf(file_path)
    chunks = chunk_past_paper(pages, paper_year=year)
    if not chunks:
        raise ValueError(
            f"No extractable text found in the paper for {year} "
            f"({os.path.basename(file_path)}) — it may be a scanned/image-only "
            "PDF with no text layer, or its content couldn't be split into questions."
        )
    client = get_chroma_client()
    collection_name = f"papers_{subject.replace(' ', '_').lower()}"

    try:
        collection = client.get_collection(collection_name)
    except Exception:
        collection = client.create_collection(collection_name)

    try:
        collection.delete(where={"year": year})
    except Exception:
        pass

    texts = [c["text"] for c in chunks]
    embeddings = EMBED_MODEL.encode(texts).tolist()
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [{"year": c["year"], "question_num": str(c["question_num"]), "source": "past_paper"} for c in chunks]

    collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)
    return len(chunks)