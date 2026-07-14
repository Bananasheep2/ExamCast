"""Chunk-granularity sweep for the ExamCast slides_st2131 retrieval eval.

Re-ingests the SAME slide corpus under several chunking configs into a
*separate* throwaway Chroma store (never the production ./chroma_db), re-maps
the gold labels to each config's chunk ids by page overlap, and runs the real
eval/run_eval.py against every config. The embedding model and retrieval path
are held fixed — chunking is the only variable.

Gold-label re-mapping (faithful, no silent drops)
--------------------------------------------------
Each canonical gold chunk "0_slides_N" covers a known set of source pages (its
3-page baseline window). For a finer config, the gold label maps to EVERY new
chunk whose pages intersect those gold pages — so a split gold chunk is covered
by all of its sub-chunks rather than dropped. This is generous by construction
(see the printed caveat): finer configs get more gold targets per query.

Run:  PYTHONPATH=. venv/bin/python eval/chunk_sweep.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ART = os.path.join(HERE, "artifacts")
PAGES_JSON = os.path.join(ART, "pages_st2131.json")
CANON_EVAL = os.path.join(HERE, "rag_eval_set.json")
SCRATCH_DB = os.path.join(ART, "sweep_chroma")

sys.path.insert(0, ROOT)
from pipeline.ingest import EMBED_MODEL  # noqa: E402
from utils.pdf_utils import chunk_slides, chunk_slides_by_tokens  # noqa: E402
import run_eval  # eval/ is on sys.path when invoked as a script  # noqa: E402
import chromadb  # noqa: E402

FILE_PREFIX = "0_"  # matches ingest_slides' "{file_idx}_" chunk-id prefix

# Configs to sweep. label -> (builder, human-readable size). Builders take the
# page list and return chunk dicts (chunk_id, text, pages, source).
CONFIGS = [
    ("page3_baseline", "3 pages",   lambda pg: chunk_slides(pg, 3, 0)),
    ("page2",          "2 pages",   lambda pg: chunk_slides(pg, 2, 0)),
    ("tok900_90",      "~900 tok",  lambda pg: chunk_slides_by_tokens(pg, EMBED_MODEL.tokenizer, 900, 90)),
    ("page1",          "1 page",    lambda pg: chunk_slides(pg, 1, 0)),
    ("tok512_64",      "512 tok",   lambda pg: chunk_slides_by_tokens(pg, EMBED_MODEL.tokenizer, 512, 64)),
    ("tok256_32",      "256 tok",   lambda pg: chunk_slides_by_tokens(pg, EMBED_MODEL.tokenizer, 256, 32)),
]


def load_pages():
    with open(PAGES_JSON) as f:
        return json.load(f)


def canonical_gold_pages(pages):
    """gold chunk id ("0_slides_N") -> set of source page numbers it covers,
    derived from the 3-page baseline chunking (the granularity gold was written
    against). Also returns page -> ground-truth chapter."""
    base = chunk_slides(pages, 3, 0)
    gold_pages = {FILE_PREFIX + c["chunk_id"]: set(c["pages"]) for c in base}
    # ground-truth chapter per page, via the baseline index-based chapter_of
    assert not run_eval.CHAPTER_MAP, "run sweep without EXAMCAST_CHAPTER_MAP set"
    page_chapter = {}
    for c in base:
        ch = run_eval.chapter_of(FILE_PREFIX + c["chunk_id"])
        for p in c["pages"]:
            page_chapter[p] = ch
    return gold_pages, page_chapter


def build_config(label, builder, pages, gold_pages, page_chapter, spec):
    chunks = builder(pages)
    for c in chunks:
        c["chunk_id"] = FILE_PREFIX + c["chunk_id"]

    # chunk -> pages, and chapter (chapter of its first page)
    chunk_pages = {c["chunk_id"]: set(c["pages"]) for c in chunks}
    chapter_map = {c["chunk_id"]: page_chapter[min(c["pages"])] for c in chunks}

    # ingest into an isolated collection in the scratch store
    coll_name = f"slides_st2131__{label}"
    client = chromadb.PersistentClient(path=SCRATCH_DB)
    try:
        client.delete_collection(coll_name)
    except Exception:
        pass
    coll = client.create_collection(coll_name)
    texts = [c["text"] for c in chunks]
    embs = EMBED_MODEL.encode(texts).tolist()
    coll.add(documents=texts, embeddings=embs,
             ids=[c["chunk_id"] for c in chunks],
             metadatas=[{"pages": str(sorted(c["pages"]))} for c in chunks])

    # re-map gold: union of gold pages -> every chunk intersecting them
    remapped_q = []
    remap_notes = []
    for q in spec["questions"]:
        want_pages = set()
        for g in q["gold_chunks"]:
            want_pages |= gold_pages[g]
        new_gold = sorted(
            (cid for cid, pgs in chunk_pages.items() if pgs & want_pages),
            key=lambda x: int(x.split("_")[-1]),
        )
        assert new_gold, f"{label}: query {q['id']} lost all gold — pages {want_pages}"
        if len(new_gold) != len(q["gold_chunks"]):
            remap_notes.append((q["id"], len(q["gold_chunks"]), len(new_gold)))
        nq = dict(q)
        nq["gold_chunks"] = new_gold
        remapped_q.append(nq)

    per_spec = dict(spec)
    per_spec["collection"] = coll_name
    per_spec["questions"] = remapped_q

    eval_path = os.path.join(ART, f"evalset_{label}.json")
    chap_path = os.path.join(ART, f"chapters_{label}.json")
    with open(eval_path, "w") as f:
        json.dump(per_spec, f)
    with open(chap_path, "w") as f:
        json.dump(chapter_map, f)

    return {
        "label": label,
        "n_chunks": len(chunks),
        "avg_pages": sum(len(p) for p in chunk_pages.values()) / len(chunks),
        "avg_gold": sum(len(q["gold_chunks"]) for q in remapped_q) / len(remapped_q),
        "eval_path": eval_path,
        "chap_path": chap_path,
        "remap_notes": remap_notes,
    }


def run_eval_subprocess(cfg):
    out_path = os.path.join(ART, f"summary_{cfg['label']}.json")
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": ROOT,
        "EXAMCAST_CHROMA_PATH": SCRATCH_DB,
        "EXAMCAST_EVAL_COLLECTION": f"slides_st2131__{cfg['label']}",
        "EXAMCAST_EVAL_SET": cfg["eval_path"],
        "EXAMCAST_CHAPTER_MAP": cfg["chap_path"],
        "EXAMCAST_EVAL_OUT": out_path,
    })
    subprocess.run([sys.executable, os.path.join(HERE, "run_eval.py")],
                   env=env, cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(out_path) as f:
        return json.load(f)


def main():
    os.makedirs(ART, exist_ok=True)
    pages = load_pages()
    with open(CANON_EVAL) as f:
        spec = json.load(f)
    gold_pages, page_chapter = canonical_gold_pages(pages)

    rows = []
    for label, size, builder in CONFIGS:
        cfg = build_config(label, builder, pages, gold_pages,
                           page_chapter, spec)
        summ = run_eval_subprocess(cfg)
        cfg["summary"] = summ
        cfg["size"] = size
        rows.append(cfg)

    # ---- comparison table ----
    print("\n" + "=" * 118)
    print("CHUNK-GRANULARITY SWEEP  —  slides_st2131  (N=30 queries, "
          "embed=all-MiniLM-L6-v2 fixed, generous page-overlap gold remap)")
    print("=" * 118)
    hdr = (f"{'config':<16}{'chunk size':<12}{'n_chunks':<10}"
           f"{'hit@1':<8}{'hit@3':<8}{'hit@5':<8}"
           f"{'rec@1':<8}{'rec@3':<8}{'rec@5':<8}"
           f"{'adj-sub@1':<11}{'partial@5':<10}{'gold/q':<7}")
    print(hdr)
    print("-" * 118)
    for cfg in rows:
        m = cfg["summary"]["metrics"]
        print(f"{cfg['label']:<16}{cfg['size']:<12}{cfg['n_chunks']:<10}"
              f"{m['1']['hit_rate']:<8.3f}{m['3']['hit_rate']:<8.3f}{m['5']['hit_rate']:<8.3f}"
              f"{m['1']['recall']:<8.3f}{m['3']['recall']:<8.3f}{m['5']['recall']:<8.3f}"
              f"{m['1']['adjacent_chunk_substitution']:<11}"
              f"{m['5']['partial_coverage']:<10}{cfg['avg_gold']:<7.2f}")
    print("-" * 118)
    print("adj-sub@1 = adjacent-chunk-substitution failures at k=1 (dominant baseline mode).")
    print("partial@5 = multi-gold queries that hit but missed some gold at k=5.")
    print("gold/q    = mean gold chunks per query AFTER remap (>1.0 => generous inflation).")

    # ---- remap report ----
    print("\n" + "-" * 118)
    print("GOLD RE-MAP REPORT  (queries whose gold-chunk count changed vs the 3-page baseline)")
    print("-" * 118)
    for cfg in rows:
        if not cfg["remap_notes"]:
            print(f"{cfg['label']:<16} none (gold count unchanged for all 30 queries)")
            continue
        notes = ", ".join(f"{qid}:{a}->{b}" for qid, a, b in cfg["remap_notes"])
        print(f"{cfg['label']:<16} {len(cfg['remap_notes'])} remapped  |  {notes}")

    # save full roll-up
    with open(os.path.join(ART, "sweep_results.json"), "w") as f:
        json.dump([{k: v for k, v in c.items() if k != "summary"} | {"summary": c["summary"]}
                   for c in rows], f, indent=2)
    print("\nFull roll-up: eval/artifacts/sweep_results.json")


if __name__ == "__main__":
    main()
