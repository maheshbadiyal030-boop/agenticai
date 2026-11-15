from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from config import HISTORY_PATH


@dataclass
class InterviewMemory:
    session_id: str = field(default_factory=lambda: str(uuid4()))
    company: str = ""
    job_role: str = ""
    interview_type: str = "Mixed"
    starting_difficulty: str = "Medium"
    number_of_questions: int = 5
    entries: list[dict[str, Any]] = field(default_factory=list)
    current_question: dict[str, Any] | None = None

    @property
    def is_complete(self) -> bool:
        return len(self.entries) >= self.number_of_questions

    @property
    def current_difficulty(self) -> str:
        if len(self.entries) < 2:
            return self.starting_difficulty
        recent_scores = [entry["evaluation"]["total_score"] for entry in self.entries[-2:]]
        average = sum(recent_scores) / len(recent_scores)
        levels = ["Easy", "Medium", "Hard"]
        index = levels.index(self.starting_difficulty)
        if average >= 32:
            return levels[min(index + 1, len(levels) - 1)]
        if average <= 18:
            return levels[max(index - 1, 0)]
        return self.starting_difficulty


def load_session_history() -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    with HISTORY_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("The local session history must contain a JSON list.")
    return data


def save_completed_session(memory: InterviewMemory, report: dict[str, Any]) -> None:
    history = load_session_history()
    if any(item.get("session_id") == memory.session_id for item in history):
        return
    history.append(
        {
            "session_id": memory.session_id,
            "company": memory.company,
            "job_role": memory.job_role,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "overall_score": report["overall_score"],
            "entries": memory.entries,
            "readiness": report["readiness"],
        }
    )
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = HISTORY_PATH.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)
    temporary_path.replace(HISTORY_PATH)
