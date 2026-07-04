import json
import re
import google.generativeai as genai
import os
from dotenv import load_dotenv
from pydantic import BaseModel

from pipeline.ingest import get_chroma_client
from pipeline.llm_client import get_model

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# ── Pydantic models for structured output ──────────────────


class SubPart(BaseModel):
    label: str  # e.g. "a", "b"
    text: str
    marks: int


class Question(BaseModel):
    model_config = {"protected_namespaces": ()}
    question_number: int
    question_text: str
    marks: int
    subparts: list[SubPart] = []
    model_answer: list[str] = []  # 3-5 bullet points


class PracticePaper(BaseModel):
    subject: str
    total_marks: int
    time_allowed: str  # e.g. "3 hours"
    questions: list[Question]


# ── Main generator function ────────────────────────────────


def generate_practice_paper(
    subject: str,
    high_prob_topics: list[dict],
    untested_topics: list[dict],
    style: dict,
    num_questions: int = 5,
    has_slides: bool = True,
) -> tuple[str, list[dict]]:
    """Generate a practice exam paper using Gemini 2.5 Flash.

    Args:
        subject: Name of the subject.
        high_prob_topics: Top topic dicts from the analyzer.
        untested_topics: Never-tested topic dicts.
        style: Exam style dict from style_extractor.
        num_questions: Target number of questions.
        has_slides: If True, lecture-notes content is injected as extra
                    grounding (notation/terminology) for the generated
                    questions. Topics always come from ``high_prob_topics``
                    (derived from the past papers), never from the notes.

    Returns:
        Tuple of (raw_text, structured_questions).
        - raw_text: The raw Gemini response (for backward compat / text download).
        - structured_questions: list[dict] of Question objects, or a fallback
          single-question list if JSON parsing fails.
    """
    client_chroma = get_chroma_client()
    subject_key = subject.replace(" ", "_").lower()

    # ── Assemble topic context (always derived from the past papers) ──
    topic_context_parts = []
    for topic in high_prob_topics:
        years = topic.get("years", [])
        count = topic.get("count", 0)
        content = topic.get("content", "")
        topic_context_parts.append(
            f"[Topic — tested {count}x in years {years}]\n{content}"
        )
    # Untested topics are typically empty (papers-only analysis) but honored.
    for topic in untested_topics[:2]:
        content = topic.get("content", topic.get("preview", ""))
        if content:
            topic_context_parts.append(
                "[Topic — NOT YET TESTED, potential surprise]\n" + content
            )

    topic_context = "\n\n".join(topic_context_parts)

    # ── Optional lecture-notes grounding ───────────────────
    # Lecture notes never decide the topics (those come from the papers); when
    # present they only give the model real course notation/terminology so the
    # generated questions stay faithful to the actual material.
    notes_block = ""
    if has_slides:
        try:
            slides_collection = client_chroma.get_collection(f"slides_{subject_key}")
            slide_docs = slides_collection.get(include=["documents"])["documents"]
            notes_context = "\n\n".join(slide_docs[:8])[:4000]
        except Exception:
            notes_context = ""
        if notes_context:
            notes_block = (
                "\n=== COURSE NOTES (reference only — use for correct notation and "
                "terminology; do NOT draw new topics from here) ===\n"
                f"{notes_context}\n"
            )

    # ── Few-shot examples from past papers ─────────────────
    papers_collection = client_chroma.get_collection(f"papers_{subject_key}")
    sample_questions = papers_collection.get(include=["documents"])
    few_shot_examples = "\n\n".join(sample_questions["documents"][:3])

    # ── Style description block ────────────────────────────
    style_desc = f"""\
- Common question verbs: {', '.join(style.get('common_verbs', ['discuss', 'explain']))}
- Question format: {style.get('question_format', 'mixed')}
- Sub-part structure: {style.get('subpart_pattern', 'typically (a)(b)(c)')}
- Marks per question: {style.get('marks_per_question', '10-20 marks')}
- Total marks: {style.get('estimated_total_marks', 100)}
- Number of sections: {style.get('total_sections', 3)}
- Question phrasing style: {style.get('sample_question_style', 'academic and formal')}"""

    # ── Build the prompt ───────────────────────────────────
    prompt = f"""You are generating a practice exam paper for the subject: {subject}

CRITICAL CONSTRAINT: You may ONLY write questions about topics explicitly listed in the TOPIC CONTENT section below. Do not introduce any concept not present in that section.

=== TOPIC CONTENT (your only allowed source of topics) ===
{topic_context}
{notes_block}
=== STYLE REQUIREMENTS ===
{style_desc}

=== REAL PAST PAPER EXAMPLES (match this style exactly) ===
{few_shot_examples}

=== YOUR TASK ===
Generate a complete practice exam paper with {num_questions} questions.

IMPORTANT: Do NOT use LaTeX notation (no \\textbf{{}}, \\frac{{}}{{}}, \\text{{}}, $...$, \\[...\\], etc.). Use plain text with ^ for powers and _ for subscripts. A single digit exponent/subscript needs no braces (e.g., X^2, X_1), but ANY exponent or subscript with more than one character — including a sign, a letter, or an expression — MUST be wrapped in curly braces (e.g., X^{{-1}}, e^{{-x}}, X_{{i}}, X_{{n-1}}). Never use parentheses for powers or subscripts (e.g. do NOT write e^(-x); write e^{{-x}}).

For model answers, write DETAILED STEP-BY-STEP explanations (4-6 steps each). Each step should explain the reasoning, formula, or calculation performed — not just state a fact. Write as if teaching a student how to solve the problem.

CRITICAL for subparts: If a question has subparts (a, b, c), the "question_text" must be the STEM only (the preamble before any sub-questions). Put each sub-question EXCLUSIVELY in the "subparts" array. Do NOT repeat a), b) labels inside question_text — this causes duplication in the output.

Return ONLY a valid JSON object (no markdown fences, no backticks, no preamble) with this EXACT structure:
{{
  "subject": "{subject}",
  "total_marks": {style.get('estimated_total_marks', 100)},
  "time_allowed": "X hours",
  "questions": [
    {{
      "question_number": 1,
      "question_text": "Full question text in plain English...",
      "marks": 15,
      "subparts": [
        {{"label": "a", "text": "sub-question text", "marks": 5}},
        {{"label": "b", "text": "sub-question text", "marks": 10}}
      ],
      "model_answer": [
        "Step 1: Identify given parameters and what the question asks for (e.g., we are given X ~ Exp(1) and need the distribution of Y = 1/X^2).",
        "Step 2: Determine the appropriate method. Since Y is a function of X, choose the transformation method (Jacobian) or the CDF method.",
        "Step 3: Apply the chosen method. For CDF: F_Y(y) = P(Y <= y) = P(1/X^2 <= y) = P(X >= 1/sqrt(y)).",
        "Step 4: Compute using the known distribution. For Exp(1): P(X >= a) = e^(-a), so F_Y(y) = e^(-1/sqrt(y)) for y > 0.",
        "Step 5: Differentiate the CDF to obtain the PDF. f_Y(y) = d/dy F_Y(y) = ...",
        "Step 6: State the final answer with the support (range of y) clearly indicated."
      ]
    }}
  ]
}}

Match the real past paper style in phrasing, difficulty, and structure. Show marks in brackets after each question.

Generate the JSON now:"""

    model = get_model("gemini-3.1-flash-lite", mock_key="paper")
    response = model.generate_content(prompt)
    raw_text = response.text

    # ── Parse structured JSON from the response ────────────
    structured = _parse_structured_output(raw_text, style, subject)

    return raw_text, structured


# ── JSON parsing helpers ───────────────────────────────────


def _parse_structured_output(
    raw_text: str, style: dict, subject: str
) -> list[dict]:
    """Try to parse Gemini's response as structured JSON.

    Returns a list of question dicts on success, or a fallback
    single-question list on failure.
    """
    try:
        # Strip common markdown fence wrappers
        cleaned = raw_text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        data = json.loads(cleaned)
        paper = PracticePaper(**data)
        return [q.model_dump() for q in paper.questions]

    except (json.JSONDecodeError, Exception):
        # Fallback: wrap the raw text as a single pseudo-question
        return [
            {
                "question_number": 1,
                "question_text": raw_text,
                "marks": style.get("estimated_total_marks", 100),
                "subparts": [],
                "model_answer": ["See generated text above."],
            }
        ]
