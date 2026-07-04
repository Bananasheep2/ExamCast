"""Concurrency / cross-run data-integrity tests (Phase 1 table: ST-1, ST-2).

These are the highest-priority findings from the Phase 1 audit: chroma_db
collections are keyed only by subject name, globally shared, with no
per-session isolation and no locking around delete+recreate. Both tests
below reproduce real, verified corruption/loss against the actual
ingest_slides / ingest_past_paper functions — not a reimplementation.
"""
import threading

from pipeline.ingest import ingest_slides, ingest_past_paper


# ── ST-1: concurrent users on the same subject name race and lose data ──


def test_st1_concurrent_slides_ingest_for_same_subject_does_not_lose_data(isolated_chroma):
    """ST-1: two (or more) sessions uploading slides for the SAME subject
    name concurrently race on ingest_slides' delete_collection() +
    create_collection() pair, which is not atomic. Verified directly: only
    1 of 3 concurrent workers survives, the other 2 raise uncaught
    UniqueConstraintError / InvalidCollectionException and their content
    never makes it into the collection at all.

    Expected to FAIL today — asserts the desired outcome (all 3 sources'
    content present), not the current one."""
    subject = "ST1 Concurrent Slides Subject"
    sources = [
        "tests/fixtures/2021_paper.pdf",
        "tests/fixtures/paper_2022.pdf",
        "tests/fixtures/paper_2023.pdf",
    ]
    results = {}
    errors = []
    lock = threading.Lock()

    def worker(idx, path):
        try:
            n = ingest_slides(path, subject)
            with lock:
                results[idx] = n
        except Exception as e:
            with lock:
                errors.append((idx, type(e).__name__, str(e)))

    threads = [threading.Thread(target=worker, args=(i, p)) for i, p in enumerate(sources)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, (
        f"concurrent ingest for the same subject raised uncaught errors "
        f"instead of being safely serialized: {errors}"
    )
    assert len(results) == len(sources), (
        f"expected all {len(sources)} concurrent sources to be ingested, "
        f"only {len(results)} succeeded — the rest silently lost their data"
    )


def test_st1_concurrent_papers_ingest_different_years_same_subject(isolated_chroma):
    """ST-1 sibling case: concurrent paper ingests for DIFFERENT years under
    the SAME subject. Uses get_or_create-shaped logic (less destructive than
    the slides path's delete+recreate), so this races less reliably than
    the slides test above — but it IS the same unlocked TOCTOU pattern and
    WILL intermittently fail (confirmed: passed on first authoring run,
    failed on a later CI run with no code changes in between). This is not
    a flaky test to quarantine — it's the same unresolved ST-1 root cause
    surfacing nondeterministically. Listed as a known-unresolved case."""
    subject = "ST1 Concurrent Papers Subject"
    jobs = [
        ("2021", "tests/fixtures/2021_paper.pdf"),
        ("2022", "tests/fixtures/paper_2022.pdf"),
    ]
    errors = []
    lock = threading.Lock()

    def worker(year, path):
        try:
            ingest_past_paper(path, subject, year)
        except Exception as e:
            with lock:
                errors.append((year, type(e).__name__, str(e)))

    threads = [threading.Thread(target=worker, args=(y, p)) for y, p in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent paper ingest raised uncaught errors: {errors}"

    subject_key = subject.replace(" ", "_").lower()
    col = isolated_chroma.get_collection(f"papers_{subject_key}")
    years_present = {m.get("year") for m in col.get(include=["metadatas"])["metadatas"]}
    assert years_present == {"2021", "2022"}, f"expected both years, got {years_present}"


# ── ST-2: sequential re-run of the same subject leaks stale data ───────


def test_st2_rerun_with_different_papers_does_not_leak_stale_years(isolated_chroma):
    """ST-2: a single user generates once with papers from 2021+2022, then
    changes their mind and re-runs the SAME subject with only 2023. The
    'fresh' analysis should reflect only 2023 — instead, ingest_past_paper
    only deletes the specific year being re-uploaded, so 2021/2022 persist
    forever and silently contaminate the new run.

    Expected to FAIL today — asserts the desired (fresh) outcome."""
    subject = "ST2 Rerun Subject"

    # Run 1: user uploads 2021 + 2022
    ingest_past_paper("tests/fixtures/2021_paper.pdf", subject, "2021")
    ingest_past_paper("tests/fixtures/paper_2022.pdf", subject, "2022")

    # Run 2 (same subject, "Start over" then re-generate): user uploads ONLY 2023
    ingest_past_paper("tests/fixtures/paper_2023.pdf", subject, "2023")

    subject_key = subject.replace(" ", "_").lower()
    col = isolated_chroma.get_collection(f"papers_{subject_key}")
    years_present = {m.get("year") for m in col.get(include=["metadatas"])["metadatas"]}

    assert years_present == {"2023"}, (
        f"expected only the years from the most recent run ({{'2023'}}), but "
        f"found stale data from a prior run still present: {years_present}"
    )
