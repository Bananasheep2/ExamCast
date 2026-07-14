"""Ground-truth retrieval eval for the ExamCast RAG pipeline.

Measures the real retrieval path used in pipeline/analyzer.py: embed a query
with all-MiniLM-L6-v2 and query the slides_<subject> Chroma collection.

Reports:
  * hit-rate@k  -- fraction of queries with >=1 gold chunk in the top-k
  * recall@k    -- mean over queries of |gold ∩ top-k| / |gold|
  * a breakdown of failures into distinct failure modes.

Run:  python eval/run_eval.py
"""
import json
import os
from collections import Counter

from pipeline.ingest import EMBED_MODEL, get_chroma_client

HERE = os.path.dirname(os.path.abspath(__file__))
# EXAMCAST_EVAL_SET lets the chunk-granularity sweep point at a per-config eval
# set whose gold labels are re-mapped to that config's chunk ids. Unset in
# normal operation, so the default is the canonical 3-page eval set.
EVAL_SET = os.getenv("EXAMCAST_EVAL_SET") or os.path.join(HERE, "rag_eval_set.json")

# Deep window used only to locate where a missed gold chunk actually ranked,
# so we can tell a crowded-out near-miss from a true semantic miss.
DEEP_N = 15

# Optional {chunk_id -> chapter} override. The default chapter_of() below keys
# off the 3-page baseline chunk index; under a different chunk granularity that
# index no longer means the same pages, so the sweep supplies a page-derived
# map via EXAMCAST_CHAPTER_MAP to keep the adjacent-chunk-substitution heuristic
# faithful. Empty/unset => original index-based behaviour.
_chapter_map_path = os.getenv("EXAMCAST_CHAPTER_MAP")
CHAPTER_MAP: dict = {}
if _chapter_map_path:
    with open(_chapter_map_path) as _f:
        CHAPTER_MAP = json.load(_f)

# Chapter/section a slide chunk belongs to, keyed by its slide index
# (chunk id "0_slides_N" -> N). Lets us detect when the retriever found the
# right topic area but the wrong specific chunk ("adjacent-chunk substitution").
def chapter_of(chunk_id: str) -> str:
    if CHAPTER_MAP:
        return CHAPTER_MAP.get(chunk_id, "unknown")
    try:
        n = int(chunk_id.split("_")[-1])
    except ValueError:
        return "unknown"
    if n == 0:
        return "cover"
    if 1 <= n <= 7:
        return "ch1-combinatorics"
    if 8 <= n <= 9:
        return "ch3-conditional"
    if 10 <= n <= 16:
        return "ch4-discrete"
    if 17 <= n <= 23:
        return "ch5-continuous"
    if 24 <= n <= 27:
        return "ch6-joint"
    if 28 <= n <= 33:
        return "ch7-expectation"
    if n == 34:
        return "ch8-clt"
    return "unknown"


def rank_of_gold(ranked_ids, gold):
    """1-based rank of the best-placed gold chunk, or None if outside window."""
    best = None
    for g in gold:
        if g in ranked_ids:
            r = ranked_ids.index(g) + 1
            best = r if best is None else min(best, r)
    return best


def classify_hit_failure(gold, top_ids, ranked_ids, k):
    """Bucket a query that retrieved ZERO gold chunks in the top-k."""
    gold_chapters = {chapter_of(g) for g in gold}
    top_chapters = {chapter_of(c) for c in top_ids}
    # Priority 1: retriever surfaced the right chapter but the wrong chunk.
    if gold_chapters & top_chapters:
        return "adjacent-chunk-substitution"
    best = rank_of_gold(ranked_ids, gold)
    # Priority 2: gold ranked just outside top-k (<= deep window) -> crowded out.
    if best is not None and best <= DEEP_N:
        return "distractor-domination"
    # Priority 3: gold ranked below the deep window -> genuine embedding miss.
    return "semantic-gap"


def evaluate(questions, ranked_by_id, k):
    """Score all queries at cutoff k. Returns metrics, per-query rows, and
    failure counters (hit-failures by mode + multi-gold partial-coverage)."""
    hits = 0
    recall_sum = 0.0
    rows = []
    hit_failures = Counter()
    recall_failures = 0

    for q in questions:
        gold = q["gold_chunks"]
        ranked_ids = ranked_by_id[q["id"]]
        top_ids = ranked_ids[:k]

        found = [g for g in gold if g in top_ids]
        is_hit = len(found) > 0
        recall = len(found) / len(gold)
        hits += int(is_hit)
        recall_sum += recall

        mode = None
        if not is_hit:
            mode = classify_hit_failure(gold, top_ids, ranked_ids, k)
            hit_failures[mode] += 1
        elif recall < 1.0:
            recall_failures += 1
            mode = "partial-coverage"

        rows.append({"id": q["id"], "hit": is_hit, "recall": recall,
                     "gold_rank": rank_of_gold(ranked_ids, gold), "mode": mode})

    n = len(questions)
    return {
        "k": k, "hit_rate": hits / n, "hits": hits, "n": n,
        "recall": recall_sum / n, "rows": rows,
        "hit_failures": hit_failures, "recall_failures": recall_failures,
    }


def main():
    with open(EVAL_SET) as f:
        spec = json.load(f)

    primary_k = spec["k"]
    # EXAMCAST_EVAL_COLLECTION lets the sweep evaluate a per-config collection
    # without editing the eval set. Defaults to the set's declared collection.
    collection_name = os.getenv("EXAMCAST_EVAL_COLLECTION") or spec["collection"]
    collection = get_chroma_client().get_collection(collection_name)
    questions = spec["questions"]

    # Embed + retrieve once per query at the deep window; every cutoff k is a
    # prefix of this ranking, so the retrieval path is identical across k.
    ranked_by_id = {}
    for q in questions:
        emb = EMBED_MODEL.encode([q["query"]]).tolist()
        res = collection.query(query_embeddings=emb, n_results=DEEP_N,
                               include=["distances"])
        ranked_by_id[q["id"]] = res["ids"][0]

    n = len(questions)
    ks = sorted({1, 3, primary_k})
    results = {k: evaluate(questions, ranked_by_id, k) for k in ks}

    print(f"\n{'='*68}\nRAG RETRIEVAL EVAL  —  {collection_name}  (N={n})\n{'='*68}")
    print(f"{'k':<6}{'hit-rate@k':<14}{'recall@k':<12}{'hit-failures':<14}partial")
    for k in ks:
        r = results[k]
        star = "  <- primary" if k == primary_k else ""
        print(f"{k:<6}{f'{r['hits']}/{n} = {r['hit_rate']:.3f}':<14}"
              f"{r['recall']:<12.3f}{sum(r['hit_failures'].values()):<14}"
              f"{r['recall_failures']}{star}")

    # Per-query detail at the primary cutoff.
    pr = results[primary_k]
    print(f"\n{'-'*68}\nPER-QUERY @k={primary_k}\n{'-'*68}")
    print(f"{'query id':<34}{'hit':<5}{'rec':<6}{'gold_rank':<10}mode")
    for row in pr["rows"]:
        gr = row["gold_rank"] if row["gold_rank"] is not None else f">{DEEP_N}"
        print(f"{row['id']:<34}{'Y' if row['hit'] else 'N':<5}"
              f"{row['recall']:<6.2f}{str(gr):<10}{row['mode'] or ''}")

    print(f"\n{'-'*68}\nFAILURE MODES BY CUTOFF\n{'-'*68}")
    for k in ks:
        r = results[k]
        total = sum(r["hit_failures"].values())
        print(f"\n@k={k}  —  {total} hit-failure(s), "
              f"{r['recall_failures']} partial-coverage:")
        for mode, c in r["hit_failures"].most_common():
            failed = [row["id"] for row in r["rows"] if row["mode"] == mode]
            print(f"  {mode:<30} {c:>2}   {', '.join(failed)}")
        parts = [row["id"] for row in r["rows"] if row["mode"] == "partial-coverage"]
        if parts:
            print(f"  {'partial-coverage':<30} {len(parts):>2}   {', '.join(parts)}")

    print(f"\n{'-'*68}\nLEGEND\n{'-'*68}")
    print("  adjacent-chunk-substitution : right chapter surfaced, wrong specific chunk")
    print(f"  distractor-domination       : gold ranked just outside top-k but within {DEEP_N}")
    print(f"  semantic-gap                : gold ranked below {DEEP_N} (embedding miss)")
    print("  partial-coverage            : hit, but a multi-gold query missed some gold chunks")

    # EXAMCAST_EVAL_OUT: dump a machine-readable summary so a sweep driver can
    # tabulate configs without re-parsing the printed report.
    out_path = os.getenv("EXAMCAST_EVAL_OUT")
    if out_path:
        summary = {
            "collection": collection_name,
            "n": n,
            "metrics": {
                str(k): {
                    "hit_rate": results[k]["hit_rate"],
                    "recall": results[k]["recall"],
                    "adjacent_chunk_substitution":
                        results[k]["hit_failures"].get("adjacent-chunk-substitution", 0),
                    "partial_coverage": results[k]["recall_failures"],
                    "hit_failures": dict(results[k]["hit_failures"]),
                }
                for k in ks
            },
        }
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
