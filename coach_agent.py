from __future__ import annotations

from typing import Any

from llm_client import get_chat_model
from memory_manager import InterviewMemory
from rag_pipeline import RAGPipeline


class CoachAgent:
    def __init__(self, rag: RAGPipeline, api_key: str = "") -> None:
        self.rag = rag
        self.model = get_chat_model(api_key)

    def coach(self, memory: InterviewMemory) -> dict[str, Any]:
        documents = self.rag.retrieve(
            f"{memory.company} {memory.job_role} historical feedback common mistakes",
            company=memory.company,
            job_role=memory.job_role,
            category="Mixed",
            difficulty=memory.current_difficulty,
            k=10,
        )
        context = [self.rag.parse_document(doc) for doc in documents]
        if self.model:
            prompt = (
                "Act as a practical interview coach. Give personalized, actionable "
                "suggestions based on these practice results and archive excerpts. Treat "
                "feedback as historical only when it is present in the archive; never invent "
                "company facts. Label any recommendation as your own advice. Return 3-5 "
                "concise bullet points and a short list of topics to study.\n"
                f"Company/role: {memory.company} / {memory.job_role}\n"
                f"Results: {memory.entries}\nArchive excerpts: {context}"
            )
            response = self.model.invoke(prompt)
            coaching = str(response.content).strip()
            if not coaching:
                raise ValueError("The coaching model returned an empty response.")
            return {
                "coaching": coaching,
                "topics": self._topics(memory),
                "historical_context": context,
            }
        mistakes = sorted(
            {
                row["common_mistakes"]
                for row in context
                if row.get("common_mistakes")
            }
        )
        suggestions = [
            "Use a clear structure: context, approach, result, and what you learned.",
            "Support technical claims with a concrete example and explain important trade-offs.",
            "Practice concise answers, then invite follow-up questions.",
        ]
        if mistakes:
            suggestions.append(
                "Archive-listed common mistakes (demonstration or uploaded records): "
                + "; ".join(mistakes[:4])
            )
        return {
            "coaching": "\n".join(f"- {item}" for item in suggestions),
            "topics": self._topics(memory),
            "historical_context": context,
        }

    @staticmethod
    def _topics(memory: InterviewMemory) -> list[str]:
        weak_dimensions = {}
        for entry in memory.entries:
            for dimension, score in entry["evaluation"]["scores"].items():
                weak_dimensions[dimension] = weak_dimensions.get(dimension, []) + [score]
        lowest = sorted(
            weak_dimensions,
            key=lambda key: sum(weak_dimensions[key]) / len(weak_dimensions[key]),
        )
        dimension_topics = {
            "technical_correctness": "Review core technical concepts for the selected role",
            "depth_of_knowledge": "Practice explaining concepts, trade-offs, and edge cases",
            "communication_skills": "Practice concise, structured verbal explanations",
            "problem_solving": "Work through timed problems and narrate your reasoning",
        }
        topics = [dimension_topics[key] for key in lowest[:2]]
        if not topics:
            topics.append("Review role fundamentals and practice explaining your reasoning")
        return topics
