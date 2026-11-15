from __future__ import annotations

from typing import Any

from fpdf import FPDF

from coach_agent import CoachAgent
from llm_client import get_chat_model
from memory_manager import InterviewMemory


class ReportAgent:
    def __init__(self, api_key: str = "") -> None:
        self.model = get_chat_model(api_key)

    def create_report(
        self, memory: InterviewMemory, coach_agent: CoachAgent
    ) -> dict[str, Any]:
        coaching = coach_agent.coach(memory)
        overall = sum(
            entry["evaluation"]["total_score"] for entry in memory.entries
        ) / len(memory.entries)
        dimensions = [
            "technical_correctness",
            "depth_of_knowledge",
            "communication_skills",
            "problem_solving",
        ]
        averages = {
            key: sum(entry["evaluation"]["scores"][key] for entry in memory.entries)
            / len(memory.entries)
            for key in dimensions
        }
        names = {
            "technical_correctness": "technical correctness",
            "depth_of_knowledge": "depth of knowledge",
            "communication_skills": "communication",
            "problem_solving": "problem solving",
        }
        strengths = [
            f"{names[key].capitalize()} (average {value:.1f}/10)"
            for key, value in averages.items()
            if value >= 7
        ]
        weaknesses = [
            f"{names[key].capitalize()} (average {value:.1f}/10)"
            for key, value in averages.items()
            if value < 7
        ]
        if not strengths:
            strengths = ["You completed the practice session and identified areas to improve."]
        if not weaknesses:
            weaknesses = ["No consistently low scoring dimension; continue practicing under time limits."]
        study_plan = coaching["topics"] + [
            "Complete a timed mock interview and review each answer against a clear rubric.",
            f"Practice role-specific questions for {memory.company} and {memory.job_role}.",
        ]
        readiness = (
            "Strong practice performance: continue refining consistency and interview pacing."
            if overall >= 32
            else "Developing practice performance: focus on the study plan and repeat a mock session."
            if overall >= 22
            else "Foundational practice recommended: revisit core concepts and build structured answers."
        )
        if self.model:
            prompt = (
                "Write a concise, supportive placement-readiness summary using only the "
                "provided scores and coaching. Do not make hiring guarantees or infer facts "
                "not present. Include one short summary paragraph.\n"
                f"Scores: {averages}; overall: {overall:.1f}/40\n"
                f"Coaching: {coaching['coaching']}"
            )
            response = self.model.invoke(prompt)
            summary = str(response.content).strip()
            if not summary:
                raise ValueError("The report model returned an empty summary.")
        else:
            summary = (
                f"Your average practice score was {overall:.1f}/40 for {memory.company} "
                f"{memory.job_role}. Use the focus areas and study plan below to guide your "
                "next practice session. This demonstration assessment is not a hiring prediction."
            )
        return {
            "company": memory.company,
            "job_role": memory.job_role,
            "overall_score": overall,
            "dimension_averages": averages,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "study_plan": study_plan,
            "coaching": coaching["coaching"],
            "historical_context": coaching["historical_context"],
            "readiness": readiness,
            "summary": summary,
            "entries": memory.entries,
        }

    def create_pdf(self, memory: InterviewMemory, report: dict[str, Any]) -> bytes:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, self._safe("Mock Interview Report"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(
            0, 7, self._safe(f"{memory.company} | {memory.job_role}"),
            new_x="LMARGIN", new_y="NEXT"
        )
        pdf.multi_cell(
            0, 7, self._safe(f"Overall score: {report['overall_score']:.1f}/40"),
            new_x="LMARGIN", new_y="NEXT"
        )
        pdf.multi_cell(
            0, 7, self._safe(report["summary"]),
            new_x="LMARGIN", new_y="NEXT"
        )
        pdf.ln(3)
        for heading, items in (
            ("Strengths", report["strengths"]),
            ("Focus areas", report["weaknesses"]),
            ("Personalized study plan", report["study_plan"]),
        ):
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 9, self._safe(heading), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=10)
            for item in items:
                pdf.multi_cell(
                    0, 6, self._safe(f"- {item}"), new_x="LMARGIN", new_y="NEXT"
                )
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, self._safe("Question-by-question results"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=10)
        for index, entry in enumerate(memory.entries, start=1):
            pdf.multi_cell(
                0,
                6,
                self._safe(
                    f"{index}. {entry['question']}\n"
                    f"Answer: {entry['answer']}\n"
                    f"Score: {entry['evaluation']['total_score']}/40\n"
                    f"Feedback: {entry['evaluation']['feedback']}"
                ),
                new_x="LMARGIN",
                new_y="NEXT",
            )
            pdf.ln(2)
        pdf.set_font("Helvetica", "I", 8)
        pdf.multi_cell(
            0,
            5,
            self._safe(
                "Practice feedback is for preparation only and is not a hiring prediction. "
                "Demonstration data is not verified institutional interview history."
            ),
            new_x="LMARGIN",
            new_y="NEXT",
        )
        return bytes(pdf.output())

    @staticmethod
    def _safe(text: str) -> str:
        return text.encode("latin-1", errors="replace").decode("latin-1")
