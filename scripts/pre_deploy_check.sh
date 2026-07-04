#!/usr/bin/env bash
# Pre-deploy check for exam-predictor.
#
# Runs the fast unit/integration suite, then the E2E browser suite, and
# reports results against the KNOWN set of tests that intentionally
# document unresolved bugs found in the Phase 1 edge-case audit — so
# "still red" (a known, tracked issue) is distinguished from "newly red"
# (something this change broke, or a regression). Any failure NOT in the
# known list fails the build. Each known item is annotated inline below with
# the audit case ID it maps to.
set -uo pipefail
cd "$(dirname "$0")/.."

VENV_PY="venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  VENV_PY="python3"
fi

# ── Known, tracked, currently-unresolved test failures ─────────────────
# Each maps to a named case in the Phase 1 audit table. Do NOT add to this
# list to silence a new/unrelated failure — investigate it instead.
KNOWN_UNRESOLVED=(
  "test_app_ui.py::test_fu10_duplicate_year_does_not_silently_drop_a_paper"     # FU-10
  "test_concurrency.py::test_st1_concurrent_slides_ingest_for_same_subject_does_not_lose_data"  # ST-1
  "test_concurrency.py::test_st1_concurrent_papers_ingest_different_years_same_subject"          # ST-1 (same race, less reliable trigger — intermittent by nature)
  "test_concurrency.py::test_st2_rerun_with_different_papers_does_not_leak_stale_years"          # ST-2
)
# Pre-existing failure from before this test suite existed (missing fixture
# file), unrelated to anything in Phase 1/2 — left as-is per "do not weaken
# or delete existing tests."
KNOWN_PREEXISTING=(
  "test_parser.py::test_pdf_date_takes_precedence_over_filename"
)

echo "======================================================================"
echo "  1/2  Fast unit/integration suite"
echo "======================================================================"
"$VENV_PY" -m pytest tests/ -q --tb=short 2>&1 | tee /tmp/predeploy_fast.log
FAST_STATUS=${PIPESTATUS[0]}

echo
echo "======================================================================"
echo "  2/2  E2E browser suite (Chromium / Firefox / WebKit)"
echo "======================================================================"
"$VENV_PY" -m pytest tests/test_e2e_playwright.py -m e2e -q --tb=short 2>&1 | tee /tmp/predeploy_e2e.log
E2E_STATUS=${PIPESTATUS[0]}

# ── Classify failures ────────────────────────────────────────────────
ACTUAL_FAILURES=$(grep "^FAILED " /tmp/predeploy_fast.log | sed 's/^FAILED //' | awk '{print $1}')
UNEXPECTED=()
while IFS= read -r failure; do
  [ -z "$failure" ] && continue
  short="${failure#tests/}"
  known=false
  for k in "${KNOWN_UNRESOLVED[@]}" "${KNOWN_PREEXISTING[@]}"; do
    if [[ "$short" == *"$k"* ]]; then known=true; break; fi
  done
  if [ "$known" = false ]; then
    UNEXPECTED+=("$failure")
  fi
done <<< "$ACTUAL_FAILURES"

E2E_FAILURES=$(grep "^FAILED " /tmp/predeploy_e2e.log | sed 's/^FAILED //' | awk '{print $1}')

echo
echo "======================================================================"
echo "  Summary"
echo "======================================================================"
echo "Known unresolved bugs (tracked, expected red — see PHASE2_REPORT.md): ${#KNOWN_UNRESOLVED[@]}"
for k in "${KNOWN_UNRESOLVED[@]}"; do echo "  - $k"; done
echo "Known pre-existing unrelated failure: ${#KNOWN_PREEXISTING[@]}"
for k in "${KNOWN_PREEXISTING[@]}"; do echo "  - $k"; done

FAIL_BUILD=0

if [ ${#UNEXPECTED[@]} -gt 0 ]; then
  echo
  echo "UNEXPECTED failures — not in the known list, investigate before deploying:"
  printf '  %s\n' "${UNEXPECTED[@]}"
  FAIL_BUILD=1
fi

if [ -n "$E2E_FAILURES" ]; then
  echo
  echo "E2E suite failures (should never be silently expected):"
  echo "$E2E_FAILURES" | sed 's/^/  /'
  FAIL_BUILD=1
fi

echo
if [ "$FAIL_BUILD" -eq 1 ]; then
  echo "RESULT: FAIL — unexpected regressions found."
  exit 1
else
  echo "RESULT: PASS — no unexpected regressions. ${#KNOWN_UNRESOLVED[@]} known unresolved issues remain open (not blocking, but not fixed either)."
  exit 0
fi
