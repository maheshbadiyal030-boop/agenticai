from __future__ import annotations

import os
from pathlib import Path

APP_TITLE = "Company-Smart Mock Interview Coach"
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "sample_interviews.csv"
HISTORY_PATH = PROJECT_DIR / "data" / "session_history.json"
CHROMA_PATH = PROJECT_DIR / "data" / "chroma"

REQUIRED_COLUMNS = (
    "company",
    "job_role",
    "interview_question",
    "interview_category",
    "difficulty_level",
    "expected_answer_points",
    "historical_candidate_feedback",
    "common_mistakes",
)


def _get_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st

        return str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except (FileNotFoundError, RuntimeError):
        return ""


OPENAI_API_KEY = _get_api_key()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
