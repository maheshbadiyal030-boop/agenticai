from __future__ import annotations

import re
from enum import Enum


class Intent(str, Enum):
    START = "start_interview"
    EVALUATE = "evaluate_answer"
    COACH = "receive_coaching"
    REPORT = "view_report"
    UNKNOWN = "unknown"


_PATTERNS = (
    (Intent.START, r"\b(start|begin|start a|mock interview|practice interview)\b"),
    (Intent.EVALUATE, r"\b(evaluate|grade|score|submit answer|answer evaluation)\b"),
    (Intent.COACH, r"\b(coach|coaching|suggestions|improve|study advice)\b"),
    (Intent.REPORT, r"\b(report|final results|performance summary|readiness)\b"),
)


def route_intent(request: str) -> Intent:
    normalized = re.sub(r"\s+", " ", request.casefold()).strip()
    for intent, pattern in _PATTERNS:
        if re.search(pattern, normalized):
            return intent
    return Intent.UNKNOWN
