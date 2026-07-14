# RAG retrieval eval

Ground-truth eval for the ExamCast retrieval step — the operation in
[`pipeline/analyzer.py`](../pipeline/analyzer.py) that embeds a query with
`all-MiniLM-L6-v2` and queries the `slides_<subject>` Chroma collection.

## Files
- `rag_eval_set.json` — 30 exam-style queries, each tagged with the
  `gold_chunks` (slide-chunk IDs) that genuinely cover the queried material.
  Gold labels are anchored to chunk contents via section headers and verbatim
  worked examples in `slides_st2131` (35 chunks, 3 pages each).
- `run_eval.py` — runs the real retrieval path and reports metrics + failures.

## Run
```bash
PYTHONPATH=. venv/bin/python eval/run_eval.py
```
(`venv/` is the environment with `chromadb` + `sentence-transformers`; the
`slides_st2131` collection must already be ingested in `./chroma_db`.)

## Metrics
- **hit-rate@k** — fraction of queries with ≥1 gold chunk in the top-k.
- **recall@k** — mean of `|gold ∩ top-k| / |gold|` (distinct from hit-rate only
  for the multi-gold cross-topic queries).

Reported at k = 1, 3, 5 (primary = 5) so the gradient is visible; @5 nearly
saturates on this small, topically well-separated corpus.

## Failure modes
Each failing query is bucketed into exactly one mode (priority order):
1. **adjacent-chunk-substitution** — right chapter surfaced, wrong specific chunk.
2. **distractor-domination** — gold ranked just outside top-k (within the deep
   window) — crowded out by similar chunks.
3. **semantic-gap** — gold ranked below the deep window — a genuine embedding miss.
4. **partial-coverage** — a multi-gold query hit, but missed some of its gold chunks.
