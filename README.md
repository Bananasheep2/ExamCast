# ExamCast — Predict My Exam

Upload your past exam papers and get a full practice paper — predicted topics,
matched to your exam's style, with a worked answer key — as a downloadable PDF.

ExamCast is a Streamlit web app for students revising for an exam. You give it a
subject and a few past papers (PDF); it uses Google Gemini to work out which
topics come up most, learns the paper's format and phrasing, and generates a new
practice paper in that same style. Optional lecture notes can be added to ground
the generated questions in your actual course material.

## Features

- **Topic prediction** — extracts the most-tested topics from your uploaded past
  papers (Gemini analysis of past-paper questions), with key concepts/techniques
  surfaced per topic.
- **Style profiling** — detects total marks, sections, marks-per-question,
  common command verbs, question format, and phrasing from the past papers.
- **Practice-paper generation** — produces a full paper (questions, subparts, and
  a step-by-step model answer key) matched to the predicted topics and style.
- **Optional lecture notes** — PDF/PPTX notes are used only to ground question
  wording/notation, not to pick topics.
- **Downloadable output** — renders to a PDF (with LaTeX/math cleanup) plus a
  plain-text version.
- **In-app results** — three tabs: Topic analysis, Style profile, Practice paper.
- **Local semantic store** — paper content is embedded (sentence-transformers)
  into a local ChromaDB vector store.
- **Sensible guardrails** — up to 6 past papers and 6 note files per run,
  safe handling of uploaded filenames, and a clear warning when no API key is set.

## Tech stack

- **Language:** Python (3.10+)
- **Web UI:** Streamlit
- **LLM:** Google Gemini via `google-generativeai` (model `gemini-3.1-flash-lite`)
- **Pipeline orchestration:** LangGraph (state graph: ingest → analyze → style → generate)
- **PDF/PPTX parsing:** pdfplumber (pdfminer.six), python-pptx
- **Embeddings + vector store:** sentence-transformers (`all-MiniLM-L6-v2`) + ChromaDB
- **PDF generation:** fpdf2 (+ pillow)
- **Validation:** pydantic
- **Config:** python-dotenv
- **Testing:** pytest, Playwright (e2e)

## Screenshots / demo

| Upload | Generating | Results |
| :-----: | :--------: | :-----: |
| ![Upload](https://github.com/user-attachments/assets/ad040fae-c437-4a36-b533-a186ff5694c6) | ![Generating](https://github.com/user-attachments/assets/27157c0a-e3dd-4e37-b478-aa26fb2ba083) | ![Results](https://github.com/user-attachments/assets/92f5f150-e10f-481a-a83b-b82635883461) |

## Getting started

### Prerequisites

- Python **3.10+** (developed and tested on 3.13)
- A Google Gemini API key — https://aistudio.google.com/apikey

### Install

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment variables

Copy the template and fill in your key:

```bash
cp .env.example .env
open -e .env
# open .env file and set GEMINI_API_KEY to your API key
```

| Variable | Required | Description | Example |
| --- | --- | --- | --- |
| `GEMINI_API_KEY` | Yes | Google Gemini API key. Generation is disabled (with a warning) until it's set. | `your-gemini-api-key-here` |
| `EXAMCAST_CHROMA_PATH` | No | Directory for the local ChromaDB vector store. Defaults to `./chroma_db`. | `./chroma_db` |
| `EXAMCAST_MOCK_LLM` | No (tests) | Set to `1` to return canned LLM responses instead of calling Gemini (used by the e2e suite). | `1` |

`.env` is gitignored — never commit a real key.

### Run locally

```bash
streamlit run app.py
```

### Run the tests

```bash
pip install -r requirements-dev.txt
pytest                                          # unit/integration (e2e excluded by default)

# End-to-end (launches a real Streamlit server + browsers; uses the mock LLM):
playwright install
pytest tests/test_e2e_playwright.py -m e2e
```

## Testing & Quality

- **Test suite** — the full unit/integration suite passes with **0 failures**
  (`pytest`; the slow Playwright e2e tests are excluded by default and run
  separately, see above).
- **Dependency security** — audited with
  [`pip-audit`](https://pypi.org/project/pip-audit/). **6 of the 7** flagged
  runtime dependencies are fully cleared (`langchain-core`, `langgraph`,
  `langgraph-checkpoint`, `langsmith`, `protobuf`, `python-dotenv`; `pip`
  itself was upgraded too). The 7th, **`transformers`, is intentionally held
  back**: it's a transitive dependency of `sentence-transformers==3.0.1`, which
  pins `transformers<5.0.0`, but the available CVE fixes only land in
  `transformers` 5.x — and two of its advisories have no fixed release at all.
  Upgrading would mean bumping `sentence-transformers`, changing the embedding
  model's behaviour, so it stays on the latest 4.x until that's warranted.
- **Retrieval evaluation** — [`eval/`](eval/) holds a ground-truth harness for
  the RAG retrieval step (embed a query, search the `slides_<subject>` ChromaDB
  collection). It scores a set of hand-labelled ground-truth queries with
  **hit-rate@k** and **recall@k**, buckets failures by mode, and includes a
  **chunk-granularity sweep** that re-ingests the corpus at several chunk
  sizes/overlaps to compare retrieval quality. See [`eval/README.md`](eval/README.md).

## Usage

1. Enter the **subject** name.
2. Upload **1–6 past exam papers** (PDF).
3. *(Optional)* Upload up to **6 lecture-note files** (PDF/PPTX) to ground the
   generated questions in your course material.
4. Click **Generate practice paper**.
5. Review the results across three tabs — **Topic analysis**, **Style profile**,
   **Practice paper**.
6. **Download** the paper as a PDF (or plain-text TXT).

## Deployment

This project runs locally — see [Getting started](#getting-started) below. Not
currently deployed.

`scripts/pre_deploy_check.sh` runs the unit + e2e suites as a pre-deploy gate.

## Project structure

```
exam-predictor/
├── app.py                      # Streamlit entry point (UI + screen router)
├── pipeline/                   # LangGraph pipeline
│   ├── graph.py                # pipeline definition: ingest → analyze → style → generate
│   ├── ingest.py               # PDF/PPTX extraction + ChromaDB embedding storage
│   ├── analyzer.py             # Gemini topic extraction + concept enrichment
│   ├── style_extractor.py      # Gemini exam-style profiling
│   ├── generator.py            # Gemini practice-paper generation (structured output)
│   └── llm_client.py           # Gemini model factory + mock-LLM seam
├── utils/
│   ├── pdf_utils.py            # PDF/PPTX parsing, chunking, watermark cleanup
│   └── pdf_generator.py        # fpdf2 PDF rendering + LaTeX/math cleanup
├── tests/                      # pytest suite + Playwright e2e (fixtures gitignored)
├── scripts/pre_deploy_check.sh # runs unit + e2e suites before deploy
├── requirements.txt            # runtime dependencies
├── requirements-dev.txt        # test-only dependencies
├── .env.example                # environment variable template
└── LICENSE                     # MIT
```

## Contributing

Solo project, but PRs/issues welcome. Please run `pytest` (and, for UI changes,
the e2e suite) before submitting — `scripts/pre_deploy_check.sh` runs both.

## License

MIT — see [LICENSE](LICENSE).
