"""Shared Gemini model factory with a test-only mock seam.

When EXAMCAST_MOCK_LLM=1 is set, every call site below returns a stubbed
model with a fixed, deterministic response instead of calling the real
API. This exists so the Playwright E2E test (upload -> generate -> download)
can exercise the full app mechanically without spending real Gemini quota
or being nondeterministic — it is NOT used by the app in normal operation,
and production behavior is byte-for-byte identical to before this file
existed when the env var is unset.
"""
import json
import os
from unittest.mock import MagicMock

import google.generativeai as genai

MOCK_ENV_VAR = "EXAMCAST_MOCK_LLM"

_MOCK_RESPONSES = {
    "style": json.dumps(
        {
            "common_verbs": ["Find", "Calculate", "State"],
            "question_format": "calculation",
            "subpart_pattern": "2-3 lettered subparts",
            "marks_per_question": "10-20 marks",
            "total_sections": 1,
            "estimated_total_marks": 20,
            "sample_question_style": "Mock style for E2E testing.",
            "difficulty_indicators": ["prove", "derive"],
        }
    ),
    "concepts": json.dumps(
        [
            {
                "topic_index": 0,
                "concepts": [
                    {
                        "name": "Mock Concept",
                        "description": "Mock description for E2E testing.",
                        "sample_question_index": 1,
                    }
                ],
            }
        ]
    ),
    "topics_no_slides": json.dumps(
        [
            {
                "topic_name": "Mock Topic",
                "description": "Mock topic description for E2E testing.",
                "question_count": 2,
                "years": ["2022"],
                "sample_questions": ["Mock sample question?"],
            }
        ]
    ),
    "paper": json.dumps(
        {
            "subject": "Mock Subject",
            "total_marks": 20,
            "time_allowed": "1 hour",
            "questions": [
                {
                    "question_number": 1,
                    "question_text": "Mock question for E2E testing: find X^2.",
                    "marks": 20,
                    "subparts": [],
                    "model_answer": ["Step 1: Mock answer."],
                }
            ],
        }
    ),
}


def is_mock_enabled() -> bool:
    return os.getenv(MOCK_ENV_VAR, "").strip() == "1"


def get_model(model_name: str, mock_key: str):
    """Real genai.GenerativeModel, or a canned stub for `mock_key` when
    EXAMCAST_MOCK_LLM=1. mock_key must be one of _MOCK_RESPONSES' keys."""
    if is_mock_enabled():
        mock_response = MagicMock()
        mock_response.text = _MOCK_RESPONSES[mock_key]
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        return mock_model
    return genai.GenerativeModel(model_name)
