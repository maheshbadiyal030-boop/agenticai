from __future__ import annotations

from typing import Any

import streamlit as st

from coach_agent import CoachAgent
from config import APP_TITLE, DATA_PATH, OPENAI_API_KEY, REQUIRED_COLUMNS
from data_loader import load_csv_bytes, load_records
from evaluator_agent import EvaluatorAgent
from intent_router import Intent, route_intent
from interviewer_agent import InterviewerAgent
from memory_manager import InterviewMemory, load_session_history, save_completed_session
from rag_pipeline import RAGPipeline
from report_agent import ReportAgent


st.set_page_config(page_title=APP_TITLE, page_icon="🎯", layout="wide")


def _read_archive() -> tuple[list[dict[str, str]], str]:
    uploaded = st.session_state.get("archive_bytes")
    if uploaded:
        return load_csv_bytes(uploaded), "Uploaded institutional archive"
    return load_records(DATA_PATH), "Demonstration dataset"


@st.cache_resource(show_spinner="Preparing the question archive…")
def _get_rag(records_json: str, api_key: str) -> RAGPipeline:
    import json

    return RAGPipeline(json.loads(records_json), api_key=api_key)


def _agents() -> tuple[InterviewerAgent, EvaluatorAgent, CoachAgent, ReportAgent]:
    import json

    records, _ = _read_archive()
    pipeline = _get_rag(
        json.dumps(records, ensure_ascii=False, sort_keys=True),
        OPENAI_API_KEY,
    )
    return (
        InterviewerAgent(pipeline, OPENAI_API_KEY),
        EvaluatorAgent(OPENAI_API_KEY),
        CoachAgent(pipeline, OPENAI_API_KEY),
        ReportAgent(OPENAI_API_KEY),
    )


def _ensure_state() -> None:
    if "memory" not in st.session_state:
        st.session_state.memory = InterviewMemory()
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "archive_error" not in st.session_state:
        st.session_state.archive_error = None


def _page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def _show_home(records: list[dict[str, str]]) -> None:
    st.title("🎯 Company-Smart Mock Interview Coach")
    st.markdown(
        "Practice a company- and role-specific interview, get structured feedback, "
        "and leave with a focused study plan."
    )
    st.info(
        "The included questions are demonstration data, not verified institutional "
        "interview records. Upload your institution's archive in the sidebar to use it."
    )
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("How it works")
        st.markdown(
            "1. Choose a company, role, interview style, and difficulty.\n"
            "2. Answer one question at a time.\n"
            "3. Review feedback across four scoring dimensions.\n"
            "4. Download a report and use your personalized roadmap."
        )
        if st.button("Start mock interview", type="primary", use_container_width=True):
            if route_intent("start mock interview") is not Intent.START:
                st.error("Could not route the request to interview setup.")
                return
            st.session_state.page = "setup"
            st.rerun()
    with right:
        st.subheader("Available companies and roles")
        companies: dict[str, set[str]] = {}
        for row in records:
            companies.setdefault(row["company"], set()).add(row["job_role"])
        for company, roles in sorted(companies.items()):
            st.markdown(f"**{company}** — {', '.join(sorted(roles))}")
        st.caption("Use the archive uploader on the left to add your own CSV.")


def _show_setup(records: list[dict[str, str]]) -> None:
    _page_header("Interview setup", "Choose the context for your practice session.")
    companies = sorted({row["company"] for row in records})
    if not companies:
        st.error("The loaded archive contains no companies.")
        return
    with st.form("interview_setup"):
        company = st.selectbox("Company", companies)
        roles = sorted({r["job_role"] for r in records if r["company"] == company})
        role = st.selectbox("Job role", roles)
        interview_type = st.selectbox(
            "Interview type", ["Mixed", "Technical", "HR", "Coding", "Aptitude"]
        )
        difficulty = st.selectbox("Starting difficulty", ["Easy", "Medium", "Hard"], index=1)
        number_of_questions = st.selectbox("Number of questions", [5, 10, 15])
        started = st.form_submit_button("Begin interview", type="primary")
    if started:
        memory = InterviewMemory(
            company=company,
            job_role=role,
            interview_type=interview_type,
            starting_difficulty=difficulty,
            number_of_questions=number_of_questions,
        )
        st.session_state.memory = memory
        st.session_state.page = "interview"
        try:
            interviewer, _, _, _ = _agents()
            _ask_next_question(interviewer)
        except Exception as exc:
            st.error(f"Could not start the interview: {exc}")
            st.session_state.page = "setup"
            return
        st.rerun()


def _ask_next_question(interviewer: InterviewerAgent) -> None:
    memory: InterviewMemory = st.session_state.memory
    memory.current_question = interviewer.ask_question(memory)


def _show_evaluation(result: dict[str, Any]) -> None:
    st.subheader("Answer evaluation")
    score_cols = st.columns(5)
    scores = result["scores"]
    for col, label, key in zip(
        score_cols,
        ["Technical correctness", "Depth", "Communication", "Problem solving", "Total"],
        ["technical_correctness", "depth_of_knowledge", "communication_skills", "problem_solving", "total"],
    ):
        value = result["total_score"] if key == "total" else scores[key]
        col.metric(label, f"{value}/40" if key == "total" else f"{value}/10")
    st.write(result["feedback"])
    if result["missing_concepts"]:
        st.markdown("**Concepts to add:** " + ", ".join(result["missing_concepts"]))
    if result["improved_answer"]:
        with st.expander("Suggested stronger answer"):
            st.write(result["improved_answer"])
    if result["expected_points"]:
        with st.expander("Expected answer points"):
            for point in result["expected_points"]:
                st.markdown(f"- {point}")


def _show_interview() -> None:
    memory: InterviewMemory = st.session_state.memory
    latest = st.session_state.get("latest_evaluation")
    showing_evaluation = bool(
        latest
        and memory.entries
        and memory.entries[-1]["evaluation"] == latest
    )
    if memory.is_complete and not showing_evaluation:
        st.session_state.page = "report"
        st.rerun()
    if showing_evaluation:
        _page_header(
            "Answer evaluation",
            f"{memory.company} · {memory.job_role} · Question {len(memory.entries)} of {memory.number_of_questions}",
        )
        st.progress(len(memory.entries) / memory.number_of_questions)
        st.markdown(f"### {memory.entries[-1]['question']}")
        st.markdown(f"**Your answer:** {memory.entries[-1]['answer']}")
        _show_evaluation(latest)
        if st.session_state.get("latest_coaching"):
            coaching = st.session_state.latest_coaching
            st.markdown("#### Personalized coaching")
            st.markdown(coaching["coaching"])
            st.markdown("**Topics to study:** " + ", ".join(coaching["topics"]))
        coach_col, next_col = st.columns(2)
        with coach_col:
            if st.button("Get coaching suggestions"):
                try:
                    if route_intent("receive coaching") is not Intent.COACH:
                        raise ValueError("The request could not be routed to the coach.")
                    _, _, coach, _ = _agents()
                    st.session_state.latest_coaching = coach.coach(memory)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Coaching failed: {exc}")
        with next_col:
            label = (
                "Finish interview and view report"
                if memory.is_complete
                else "Next question"
            )
            if st.button(label, type="primary"):
                if memory.is_complete:
                    st.session_state.page = "report"
                else:
                    try:
                        interviewer, _, _, _ = _agents()
                        _ask_next_question(interviewer)
                        st.session_state.latest_evaluation = None
                        st.session_state.latest_coaching = None
                    except Exception as exc:
                        st.error(f"Could not load the next question: {exc}")
                        return
                st.rerun()
        with st.expander("Conversation history", expanded=False):
            for index, entry in enumerate(memory.entries, start=1):
                st.markdown(f"**Q{index}. {entry['question']}**")
                st.markdown(f"Your answer: {entry['answer']}")
                st.caption(f"Score: {entry['evaluation']['total_score']}/40")
        return

    question = memory.current_question
    if not question:
        st.error("No question is available. Please restart the interview.")
        if st.button("Return to setup"):
            st.session_state.page = "setup"
            st.rerun()
        return

    _page_header(
        f"{memory.company} · {memory.job_role}",
        f"{memory.interview_type} interview · {memory.current_difficulty} difficulty",
    )
    st.progress(len(memory.entries) / memory.number_of_questions)
    st.caption(f"Question {len(memory.entries) + 1} of {memory.number_of_questions}")
    st.markdown(f"### {question['question']}")
    st.caption(f"{question['category']} · {question['difficulty']}")

    with st.form("answer_form", clear_on_submit=True):
        answer = st.text_area(
            "Your answer",
            height=190,
            placeholder="Explain your reasoning and include examples where relevant…",
        )
        submitted = st.form_submit_button("Submit answer", type="primary")
    if submitted:
        if not answer.strip():
            st.warning("Enter an answer before submitting.")
        else:
            try:
                intent = route_intent("evaluate answer")
                if intent is not Intent.EVALUATE:
                    raise ValueError("The answer could not be routed to the evaluator.")
                _, evaluator, _, _ = _agents()
                evaluation = evaluator.evaluate(question, answer, memory)
                memory.entries.append(
                    {
                        "question": question["question"],
                        "category": question["category"],
                        "difficulty": question["difficulty"],
                        "answer": answer.strip(),
                        "evaluation": evaluation,
                        "source": question.get("source", ""),
                    }
                )
                memory.current_question = None
                st.session_state.latest_evaluation = evaluation
                st.session_state.latest_coaching = None
            except Exception as exc:
                st.error(f"Answer evaluation failed: {exc}")
                return
            st.rerun()

    with st.expander("Conversation history", expanded=False):
        for index, entry in enumerate(memory.entries, start=1):
            st.markdown(f"**Q{index}. {entry['question']}**")
            st.markdown(f"Your answer: {entry['answer']}")
            st.caption(f"Score: {entry['evaluation']['total_score']}/40")


def _show_report() -> None:
    memory: InterviewMemory = st.session_state.memory
    if not memory.entries:
        st.warning("Complete at least one question to generate a report.")
        if st.button("Back to setup"):
            st.session_state.page = "setup"
            st.rerun()
        return
    _page_header("Final performance report", f"{memory.company} · {memory.job_role}")
    if st.session_state.get("report_session_id") != memory.session_id:
        try:
            intent = route_intent("view final report")
            if intent is not Intent.REPORT:
                raise ValueError("The request could not be routed to the report agent.")
            _, _, coach, report_agent = _agents()
            st.session_state.interview_report = report_agent.create_report(memory, coach)
            st.session_state.report_session_id = memory.session_id
        except Exception as exc:
            st.error(f"Could not generate the report: {exc}")
            return
    report = st.session_state.interview_report

    st.metric("Overall score", f"{report['overall_score']:.1f} / 40")
    st.progress(min(max(report["overall_score"] / 40, 0), 1))
    st.subheader("Placement readiness")
    st.write(report["readiness"])
    first, second = st.columns(2)
    with first:
        st.markdown("#### Strengths")
        for item in report["strengths"]:
            st.markdown(f"- {item}")
    with second:
        st.markdown("#### Focus areas")
        for item in report["weaknesses"]:
            st.markdown(f"- {item}")
    st.markdown("#### Personalized study plan")
    for index, item in enumerate(report["study_plan"], start=1):
        st.markdown(f"{index}. {item}")
    st.markdown("#### Company-specific coaching")
    st.write(report["coaching"])
    with st.expander("Question-by-question scores"):
        for index, entry in enumerate(memory.entries, start=1):
            st.markdown(f"**{index}. {entry['question']}**")
            st.write(f"Score: {entry['evaluation']['total_score']}/40")
            st.write(entry["evaluation"]["feedback"])

    try:
        pdf = report_agent.create_pdf(memory, report)
        st.download_button(
            "Download PDF report",
            data=pdf,
            file_name=f"{memory.company.lower().replace(' ', '-')}-interview-report.pdf",
            mime="application/pdf",
        )
    except Exception as exc:
        st.error(f"PDF generation failed: {exc}")
    try:
        if st.session_state.get("saved_session_id") != memory.session_id:
            save_completed_session(memory, report)
            st.session_state.saved_session_id = memory.session_id
        st.success("This completed session has been saved locally on this computer.")
    except Exception as exc:
        st.error(f"Could not save this session locally: {exc}")
    if st.button("Start another interview"):
        st.session_state.memory = InterviewMemory()
        st.session_state.latest_evaluation = None
        st.session_state.latest_coaching = None
        st.session_state.page = "setup"
        st.rerun()


def main() -> None:
    _ensure_state()
    with st.sidebar:
        st.header("Session settings")
        st.caption(
            "LLM mode: OpenAI" if OPENAI_API_KEY else "LLM mode: offline demonstration"
        )
        st.markdown("### Historical archive")
        uploaded_file = st.file_uploader(
            "Upload archive (CSV)", type=["csv"], help=f"Required columns: {', '.join(REQUIRED_COLUMNS)}"
        )
        if uploaded_file is not None:
            try:
                st.session_state.archive_bytes = uploaded_file.getvalue()
                st.session_state.archive_error = None
            except Exception as exc:
                st.session_state.archive_error = str(exc)
        if st.session_state.archive_error:
            st.error(st.session_state.archive_error)
        if st.button("Use demonstration data"):
            st.session_state.pop("archive_bytes", None)
            _get_rag.clear()
            st.rerun()
        st.markdown("---")
        if OPENAI_API_KEY:
            st.caption(
                "The Chroma index is held locally for this app session. OpenAI receives "
                "archive text for embeddings, retrieved excerpts, and student answers for "
                "AI-generated questions and feedback."
            )
        else:
            st.caption(
                "The archive and index stay local in offline demo mode; no external LLM "
                "is used. Offline scores are approximate."
            )
        with st.expander("Previous sessions"):
            try:
                history = load_session_history()
                if not history:
                    st.caption("No completed sessions saved yet.")
                for item in reversed(history[-10:]):
                    st.markdown(
                        f"**{item['company']} · {item['job_role']}**  \n"
                        f"{item['completed_at'][:10]} · {item['overall_score']:.1f}/40"
                    )
            except Exception as exc:
                st.error(f"Could not load local session history: {exc}")

    try:
        records, source_label = _read_archive()
    except Exception as exc:
        st.error(f"Unable to read the interview archive: {exc}")
        return
    st.sidebar.caption(f"Active source: {source_label} ({len(records)} records)")
    page = st.session_state.page
    if page == "home":
        _show_home(records)
    elif page == "setup":
        _show_setup(records)
    elif page == "interview":
        _show_interview()
    else:
        _show_report()


if __name__ == "__main__":
    main()
