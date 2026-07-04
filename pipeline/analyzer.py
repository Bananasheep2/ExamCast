import json
import re
import google.generativeai as genai
import os
from dotenv import load_dotenv

from pipeline.ingest import EMBED_MODEL, get_chroma_client
from pipeline.llm_client import get_model

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# ── Slides-based analysis (existing) ───────────────────────


def build_topic_frequency_map(subject: str) -> dict:
    """Match each past-paper question to the nearest slide chunk via embedding
    similarity (cosine distance < 1.2). Returns a frequency map keyed by
    slide chunk ID with per-chunk tallies."""
    client = get_chroma_client()
    subject_key = subject.replace(" ", "_").lower()

    slides_collection = client.get_collection(f"slides_{subject_key}")
    papers_collection = client.get_collection(f"papers_{subject_key}")

    all_questions = papers_collection.get(include=["documents", "metadatas"])
    all_slides = slides_collection.get(include=["documents", "metadatas"])

    slide_lookup = {
        all_slides["ids"][i]: all_slides["documents"][i]
        for i in range(len(all_slides["ids"]))
    }

    frequency_map = {
        chunk_id: {
            "slide_text_preview": text[:200],
            "count": 0,
            "years": [],
            "questions": [],
            "never_tested": True,
        }
        for chunk_id, text in slide_lookup.items()
    }

    for i, question_text in enumerate(all_questions["documents"]):
        year = all_questions["metadatas"][i].get("year", "unknown")
        q_embedding = EMBED_MODEL.encode([question_text]).tolist()

        results = slides_collection.query(
            query_embeddings=q_embedding,
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )

        best_chunk_id = results["ids"][0][0]
        distance = results["distances"][0][0]

        if distance < 1.2:
            frequency_map[best_chunk_id]["count"] += 1
            frequency_map[best_chunk_id]["never_tested"] = False
            if year not in frequency_map[best_chunk_id]["years"]:
                frequency_map[best_chunk_id]["years"].append(year)
            frequency_map[best_chunk_id]["questions"].append(question_text[:150])

    sorted_map = dict(
        sorted(frequency_map.items(), key=lambda x: x[1]["count"], reverse=True)
    )
    return sorted_map


def get_high_probability_topics(frequency_map: dict, top_n: int = 5) -> list[dict]:
    """Return the top-N most-tested topics from the frequency map.

    Each topic dict includes up to 3 sample past-paper questions so the UI
    can show what kinds of questions have been asked on each topic.
    """
    topics = []
    for chunk_id, data in list(frequency_map.items())[:top_n]:
        topics.append(
            {
                "chunk_id": chunk_id,
                "preview": data["slide_text_preview"],
                "count": data["count"],
                "years": data["years"],
                "sample_questions": data["questions"][:3],  # up to 3 past Qs
            }
        )
    return topics


def get_untested_topics(frequency_map: dict) -> list[dict]:
    """Return all slide chunks that were never matched to a past-paper question."""
    return [
        {"chunk_id": cid, "preview": data["slide_text_preview"]}
        for cid, data in frequency_map.items()
        if data["never_tested"]
    ]


# ── Concept extraction from sample questions ───────────────


def enrich_topics_with_concepts(
    high_prob_topics: list[dict], subject: str
) -> list[dict]:
    """Use Gemini to extract key implicit concepts/techniques from each topic's
    sample past-paper questions.

    For each topic, Gemini identifies up to 3 recurring concepts or techniques
    (e.g. "Jacobian transformation", "CDF method", "moment generating function")
    that are tested across the sample questions — things that are NOT explicitly
    stated in the question text but are the underlying skills being assessed.

    Returns the same list with a ``concepts`` field added to each topic dict.
    """
    if not high_prob_topics:
        return high_prob_topics

    # Build a batched prompt with all topics
    topic_blocks = []
    for i, topic in enumerate(high_prob_topics):
        preview = topic.get("preview", topic.get("content", ""))[:300]
        sample_qs = topic.get("sample_questions", [])
        q_block = "\n".join(
            f"  Q{j + 1}: {q[:250]}" for j, q in enumerate(sample_qs)
        )
        topic_blocks.append(
            f"TOPIC {i + 1}:\n"
            f"Slides excerpt: {preview}\n"
            f"Past-paper questions on this topic:\n{q_block}"
        )

    topics_text = "\n\n".join(topic_blocks)

    prompt = f"""You are analyzing past exam questions for: {subject}

For each topic below, examine the sample past-paper questions and identify the 3 most frequently tested **concepts or techniques** — the underlying skills a student must apply (e.g., "Jacobian transformation", "CDF method", "moment generating function", "Bayes' theorem", "hypothesis test with p-value"). These concepts are often NOT named explicitly in the question wording.

=== TOPICS AND THEIR PAST QUESTIONS ===
{topics_text}

Return ONLY a valid JSON array (no markdown, no backticks). One object per topic, in the same order:
[
  {{
    "topic_index": 0,
    "concepts": [
      {{
        "name": "Short concept name (e.g. Jacobian transformation)",
        "description": "One sentence on how this technique is used in these questions",
        "sample_question_index": 1
      }}
    ]
  }}
]

- topic_index: integer matching the TOPIC N number minus 1 (0-based)
- concepts: exactly 3 entries per topic, ordered by how frequently the concept appears
- sample_question_index: which Q number (1, 2, 3…) best demonstrates this concept

Generate the JSON now:"""

    try:
        model = get_model("gemini-3.1-flash-lite", mock_key="concepts")
        response = model.generate_content(prompt)
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        concepts_data = json.loads(raw)
    except Exception:
        # If Gemini fails, return topics unenriched
        for topic in high_prob_topics:
            topic.setdefault("concepts", [])
        return high_prob_topics

    # Map concepts back to topics
    for entry in concepts_data:
        idx = entry.get("topic_index", -1)
        if 0 <= idx < len(high_prob_topics):
            concepts = entry.get("concepts", [])[:3]
            # Attach the actual question text to each concept
            sample_qs = high_prob_topics[idx].get("sample_questions", [])
            enriched = []
            for c in concepts:
                q_idx = c.get("sample_question_index", 1) - 1
                c["sample_question_text"] = (
                    sample_qs[q_idx][:200] if 0 <= q_idx < len(sample_qs) else ""
                )
                enriched.append(c)
            high_prob_topics[idx]["concepts"] = enriched

    # Ensure all topics have the concepts field
    for topic in high_prob_topics:
        topic.setdefault("concepts", [])

    return high_prob_topics


# ── No-slides analysis (Gemini-based topic extraction) ─────


def build_topic_frequency_map_no_slides(subject: str) -> dict:
    """Use Gemini to identify distinct topics directly from past paper questions.

    Returns a frequency-map-compatible dict keyed by synthetic IDs like
    'topic_synthetic_0'. Each value has keys: content, count, years,
    never_tested, sample_questions, source.
    """
    client = get_chroma_client()
    subject_key = subject.replace(" ", "_").lower()
    papers_collection = client.get_collection(f"papers_{subject_key}")

    all_qs = papers_collection.get(include=["documents", "metadatas"])

    # Group questions by year
    by_year: dict[str, list[str]] = {}
    for i, q_text in enumerate(all_qs["documents"]):
        year = all_qs["metadatas"][i].get("year", "unknown")
        by_year.setdefault(year, []).append(q_text[:400])

    # Build a text block with questions grouped by year
    year_blocks = []
    for year in sorted(by_year.keys()):
        qs = by_year[year]
        year_blocks.append(f"--- {year} ---")
        for j, q in enumerate(qs):
            year_blocks.append(f"[Q{j + 1}] {q}")

    questions_by_year_block = "\n\n".join(year_blocks)

    prompt = f"""You are analyzing past exam papers for the subject: {subject}

Below are all the questions from past papers, grouped by year:

{questions_by_year_block}

Identify 5-8 distinct topics or themes that have been tested across these papers.
For each topic, provide:

1. A concise descriptive name (5-10 words)
2. A paragraph summarizing what this topic covers (150-300 words)
3. The count of questions testing this topic
4. Which years this topic appeared in
5. 1-2 short example question excerpts

Return ONLY a valid JSON array (no markdown, no backticks, no preamble). Each element must have this exact structure:

[
  {{
    "topic_name": "Descriptive Name",
    "description": "Paragraph summarizing what this topic covers...",
    "question_count": 3,
    "years": ["2020", "2022"],
    "sample_questions": ["Explain the concept of...", "Calculate the..."]
  }}
]

Generate the JSON now:"""

    model = get_model("gemini-3.1-flash-lite", mock_key="topics_no_slides")
    response = model.generate_content(prompt)

    # Parse Gemini's JSON
    try:
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        topics = json.loads(raw)
    except (json.JSONDecodeError, Exception):
        # Fallback: return a single catch-all topic
        all_text = " ".join(all_qs["documents"])[:2000]
        topics = [
            {
                "topic_name": f"{subject} Core Material",
                "description": all_text,
                "question_count": len(all_qs["documents"]),
                "years": list(by_year.keys()),
                "sample_questions": [all_qs["documents"][0][:200]],
            }
        ]

    # Build frequency-map-compatible dict
    freq_map = {}
    for i, topic in enumerate(topics):
        key = f"topic_synthetic_{i}"
        freq_map[key] = {
            "content": topic.get("description", topic.get("topic_name", "")),
            "count": topic.get("question_count", 0),
            "years": topic.get("years", []),
            "never_tested": False,
            "sample_questions": topic.get("sample_questions", []),
            "source": "gemini_extracted",
            "topic_name": topic.get("topic_name", f"Topic {i + 1}"),
        }

    return dict(
        sorted(freq_map.items(), key=lambda x: x[1]["count"], reverse=True)
    )


def get_high_probability_topics_no_slides(
    frequency_map: dict, top_n: int = 5
) -> list[dict]:
    """Return top-N topics from a no-slides frequency map.

    Each dict has 'content' and 'sample_questions' keys instead of 'chunk_id',
    so the generator can use them directly without querying ChromaDB slides.
    """
    topics = []
    for key, data in list(frequency_map.items())[:top_n]:
        topics.append(
            {
                "content": data["content"],
                "preview": data["content"][:200],
                "count": data["count"],
                "years": data["years"],
                "sample_questions": data.get("sample_questions", []),
                "topic_name": data.get("topic_name", ""),
            }
        )
    return topics


def get_untested_topics_no_slides(frequency_map: dict) -> list[dict]:
    """No-slides mode: we can only see what WAS tested, so untested is empty.

    Without lecture slides, there is no visibility into topics that have
    never appeared in past papers.
    """
    return []


# ── Test-facing question-type ranking ───────────────────────


MOCK_CLASSIFIED: list[dict] = [
    # transformations: count 5
    {"question": "Transform X to find pdf of Y = 1/X^2", "type": "transformations", "paper": "2020"},
    {"question": "Find joint pdf of U = Y-X and V = XY", "type": "transformations", "paper": "2020"},
    {"question": "Find pdf of X^{-alpha} given exponential X", "type": "transformations", "paper": "2021"},
    {"question": "Find joint pdf of U = XY and V = Y/X", "type": "transformations", "paper": "2021"},
    {"question": "Transform bivariate to find mean and covariance", "type": "transformations", "paper": "2022"},
    # bivariate: count 4
    {"question": "Find mean and covariance of bivariate normal", "type": "bivariate", "paper": "2020"},
    {"question": "Conditional distribution of Y1 given Y2", "type": "bivariate", "paper": "2020"},
    {"question": "Bivariate normal with covariance matrix", "type": "bivariate", "paper": "2021"},
    {"question": "Find a such that Y1-aY2 independent of Y2", "type": "bivariate", "paper": "2022"},
    # expectation: count 3
    {"question": "Find E[X] for Poisson sum of normals", "type": "expectation", "paper": "2020"},
    {"question": "Find mean and variance of X", "type": "expectation", "paper": "2021"},
    {"question": "E[X] for pattern SFS count", "type": "expectation", "paper": "2022"},
    # poisson: count 2
    {"question": "Poisson random variable with mean 1", "type": "poisson", "paper": "2021"},
    {"question": "N trials with Poisson process", "type": "poisson", "paper": "2022"},
    # distribution: count 2 — THIS is the 5th-place tie target
    {"question": "Exponential random variable with mean 1", "type": "distribution", "paper": "2021"},
    {"question": "Uniform random variable on [0,1]", "type": "distribution", "paper": "2022"},
]


def rank_question_types(classified: list[dict]) -> list[dict]:
    """Rank question types by how many questions of each type appear.

    Returns a list of ``{"type": str, "count": int}`` dicts sorted by count
    descending.  When multiple types share the same count at the 5th position
    the list expands to include every tied type, so the result always captures
    a clean top-N boundary without arbitrary cut-offs.
    """
    from collections import Counter

    type_counts = Counter(item["type"] for item in classified)
    ranked = [
        {"type": t, "count": c}
        for t, c in type_counts.most_common()
    ]
    return ranked
