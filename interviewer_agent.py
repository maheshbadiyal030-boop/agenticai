from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from llm_client import get_chat_model
from memory_manager import InterviewMemory
from rag_pipeline import RAGPipeline


class InterviewerAgent:
    def __init__(self, rag: RAGPipeline, api_key: str = "") -> None:
        self.rag = rag
        self.model = get_chat_model(api_key)

    def ask_question(self, memory: InterviewMemory) -> dict[str, Any]:
        query = (
            f"{memory.company} {memory.job_role} {memory.interview_type} "
            f"{memory.current_difficulty} interview question"
        )
        documents = self.rag.retrieve(
            query,
            company=memory.company,
            job_role=memory.job_role,
            category=memory.interview_type,
            difficulty=memory.current_difficulty,
            k=10,
        )
        records = [self.rag.parse_document(document) for document in documents]
        if memory.interview_type != "Mixed":
            matching = [
                row
                for row in records
                if row["interview_category"].casefold() == memory.interview_type.casefold()
            ]
            if matching:
                records = matching
        asked = {entry["question"] for entry in memory.entries}
        records = [row for row in records if row["interview_question"] not in asked]

        if self.model:
            context = "\n\n".join(self._format_record(row) for row in records[:5])
            prompt = (
                "You are a professional mock interviewer. Ask exactly one concise, "
                "role-relevant question. Use the archive context as inspiration, do not "
                "claim that any example is a verified company question. Do not include an "
                "answer or preamble. Return only the question.\n"
                f"Company: {memory.company}\nRole: {memory.job_role}\n"
                f"Interview type: {memory.interview_type}\n"
                f"Difficulty: {memory.current_difficulty}\n"
                f"Previously asked: {sorted(asked)}\nArchive context:\n{context}"
            )
            response = self.model.invoke(prompt)
            question = str(response.content).strip()
            if not question:
                raise ValueError("The interviewer model returned an empty question.")
            reference = records[0] if records else {}
            return self._make_question(
                question,
                reference,
                memory.current_difficulty,
            )

        if records:
            record = records[0]
            return self._make_question(
                record["interview_question"], record, memory.current_difficulty
            )
        category = memory.interview_type
        generic = {
            "Technical": f"How would you design and test a reliable solution to a core {memory.job_role} problem?",
            "Coding": "Describe your approach to solving a coding problem, including complexity and edge cases.",
            "HR": "Tell me about a challenging situation and how you handled it.",
            "Aptitude": "How do you break down an unfamiliar quantitative problem and verify your result?",
            "Mixed": "Describe a challenging problem relevant to this role and how you would solve it.",
        }
        return self._make_question(generic.get(category, generic["Mixed"]), {}, memory.current_difficulty)

    @staticmethod
    def _format_record(record: dict[str, str]) -> str:
        return (
            f"Category: {record.get('interview_category', '')}\n"
            f"Question: {record.get('interview_question', '')}\n"
            f"Expected points: {record.get('expected_answer_points', '')}"
        )

    @staticmethod
    def _make_question(
        question: str, record: dict[str, str], difficulty: str
    ) -> dict[str, Any]:
        return {
            "question": question,
            "category": record.get("interview_category", "General"),
            "difficulty": difficulty,
            "expected_points": [
                point.strip()
                for point in record.get("expected_answer_points", "").split(";")
                if point.strip()
            ],
            "source": record.get("source", ""),
        }
