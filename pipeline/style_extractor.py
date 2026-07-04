import google.generativeai as genai
import json
import os
from pipeline.ingest import get_chroma_client
from pipeline.llm_client import get_model
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def extract_paper_style(subject: str) -> dict:
    client_chroma = get_chroma_client()
    subject_key = subject.replace(' ', '_').lower()

    papers_collection = client_chroma.get_collection(f"papers_{subject_key}")
    all_questions = papers_collection.get(include=["documents", "metadatas"])

    sample_texts = all_questions["documents"][:10]
    sample_block = "\n\n---\n\n".join(
        f"[Question {i+1}]\n{text}"
        for i, text in enumerate(sample_texts)
    )

    prompt = f"""You are analyzing past exam papers to extract their style patterns.

Below are sample questions from past papers for the subject: {subject}

{sample_block}

Return ONLY a JSON object (no markdown, no preamble, no backticks) with exactly these keys:

{{
  "common_verbs": ["list of action verbs used in questions, e.g. discuss, compare, explain"],
  "question_format": "one of: scenario-based, abstract, calculation, mixed",
  "subpart_pattern": "description of how questions are broken into sub-parts",
  "marks_per_question": "typical range as string, e.g. 10-25 marks",
  "total_sections": 3,
  "estimated_total_marks": 100,
  "sample_question_style": "one sentence describing the typical question phrasing style",
  "difficulty_indicators": ["list of words or phrases that signal harder questions"]
}}"""

    model = get_model("gemini-3.1-flash-lite", mock_key="style")
    response = model.generate_content(prompt)
    raw_text = response.text.strip()
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    style = json.loads(raw_text)
    return style