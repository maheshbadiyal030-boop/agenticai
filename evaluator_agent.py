from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from llm_client import get_chat_model
from memory_manager import InterviewMemory


class EvaluationResult(BaseModel):
    technical_correctness: int = Field(ge=0, le=10)
    depth_of_knowledge: int = Field(ge=0, le=10)
    communication_skills: int = Field(ge=0, le=10)
    problem_solving: int = Field(ge=0, le=10)
    feedback: str
    missing_concepts: list[str]
    improved_answer: str


class EvaluatorAgent:
    def __init__(self, api_key: str = "") -> None:
        self.model = get_chat_model(api_key)

    def evaluate(
        self, question: dict[str, Any], answer: str, memory: InterviewMemory
    ) -> dict[str, Any]:
        expected = question.get("expected_points", [])
        if self.model:
            prompt = (
                "Evaluate this mock-interview answer fairly and constructively. Score each "
                "dimension from 0 to 10: technical correctness, depth of knowledge, "
                "communication skills, and problem-solving approach. Use the expected points "
                "as a rubric, not as the only acceptable answer. Be transparent about "
                "uncertainty. Return concise feedback, missing concepts, and a stronger "
                "answer outline.\n"
                f"Company/role: {memory.company} / {memory.job_role}\n"
                f"Category: {question.get('category')}\nQuestion: {question['question']}\n"
                f"Expected points: {expected}\nStudent answer: {answer}"
            )
            parsed = self.model.with_structured_output(EvaluationResult).invoke(prompt)
            scores = {
                "technical_correctness": parsed.technical_correctness,
                "depth_of_knowledge": parsed.depth_of_knowledge,
                "communication_skills": parsed.communication_skills,
                "problem_solving": parsed.problem_solving,
            }
            feedback = parsed.feedback
            missing = parsed.missing_concepts
            improved = parsed.improved_answer
        else:
            scores, feedback, missing, improved = self._offline_evaluation(answer, expected)
        return {
            "scores": scores,
            "total_score": sum(scores.values()),
            "feedback": feedback,
            "missing_concepts": missing,
            "improved_answer": improved,
            "expected_points": expected,
        }

    @staticmethod
    def _offline_evaluation(
        answer: str, expected: list[str]
    ) -> tuple[dict[str, int], str, list[str], str]:
        answer_terms = set(re.findall(r"[a-z0-9]+", answer.casefold()))
        matched = []
        missing = []
        for point in expected:
            terms = set(re.findall(r"[a-z0-9]+", point.casefold()))
            meaningful_terms = {term for term in terms if len(term) > 3}
            if meaningful_terms and len(meaningful_terms & answer_terms) / len(meaningful_terms) >= 0.25:
                matched.append(point)
            else:
                missing.append(point)
        coverage = len(matched) / max(len(expected), 1)
        length = len(answer.split())
        technical = min(10, round(2 + coverage * 8)) if expected else min(7, 3 + length // 25)
        depth = min(10, 2 + len(matched) * 2 + (1 if length > 45 else 0))
        communication = min(10, max(2, 3 + min(length // 18, 5)))
        problem_solving = min(10, 2 + len(matched) * 2 + (1 if length > 30 else 0))
        scores = {
            "technical_correctness": technical,
            "depth_of_knowledge": depth,
            "communication_skills": communication,
            "problem_solving": problem_solving,
        }
        feedback = (
            "Offline demonstration score based on answer length and keyword overlap with "
            "the archive's expected points. This is a rough practice signal, not an expert "
            "judgment of correctness."
        )
        improved = (
            "Structure your response as: direct answer, reasoning, a concrete example, "
            "trade-offs or edge cases, and a concise conclusion."
        )
        return scores, feedback, missing, improved
