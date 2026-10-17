import json
import os
import re
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai

# -------------------- Configuration --------------------
load_dotenv()
DB_PATH = "interviewace.db"
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

st.set_page_config(
    page_title="InterviewAce | AI Mock Interview",
    page_icon="🎯",
    layout="wide",
)

# -------------------- Database --------------------
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS interviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                candidate_name TEXT,
                role TEXT NOT NULL,
                experience_level TEXT NOT NULL,
                interview_type TEXT NOT NULL,
                question_count INTEGER NOT NULL,
                score REAL NOT NULL,
                feedback TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interview_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                score REAL NOT NULL,
                feedback TEXT NOT NULL,
                ideal_answer TEXT NOT NULL,
                FOREIGN KEY(interview_id) REFERENCES interviews(id)
            )
        """)

def save_interview(candidate_name, role, level, interview_type, results, overall_score, summary):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            """INSERT INTO interviews
            (created_at, candidate_name, role, experience_level, interview_type,
             question_count, score, feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                candidate_name or "Guest",
                role, level, interview_type, len(results), overall_score, summary
            ),
        )
        interview_id = cur.lastrowid
        for item in results:
            conn.execute(
                """INSERT INTO answers
                (interview_id, question, answer, score, feedback, ideal_answer)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    interview_id, item["question"], item["answer"], item["score"],
                    item["feedback"], item["ideal_answer"]
                ),
            )
        return interview_id

def load_history():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """SELECT id, created_at, candidate_name, role, experience_level,
                      interview_type, question_count, score, feedback
               FROM interviews ORDER BY id DESC""",
            conn,
        )

def load_interview_answers(interview_id):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """SELECT question, answer, score, feedback, ideal_answer
               FROM answers WHERE interview_id = ?""",
            conn,
            params=(int(interview_id),),
        )

# -------------------- Gemini helpers --------------------
def get_client():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

def extract_json(text):
    """Parse JSON from a model response, including responses wrapped in code fences."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}|\[.*\]", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("The AI response was not valid JSON. Please try again.")

def call_gemini(prompt):
    client = get_client()
    if client is None:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to your .env file and restart Streamlit."
        )
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    output = getattr(response, "text", None)
    if not output:
        raise RuntimeError("Gemini returned an empty response. Please try again.")
    return output

def generate_questions(role, level, interview_type, count, job_description=""):
    prompt = f"""
You are an experienced job interviewer. Create interview questions for practice.
Job role: {role}
Candidate level: {level}
Interview type: {interview_type}
Number of questions: {count}
Optional job description:
{job_description if job_description.strip() else "Not provided"}

Return ONLY valid JSON in exactly this format:
{{
  "questions": [
    {{"question": "question text", "topic": "topic name"}}
  ]
}}
Requirements:
- Return exactly {count} different questions.
- Match the role and experience level.
- For HR interviews, use behavioral, communication, teamwork, motivation, and workplace scenarios.
- For Technical interviews, focus on role-related knowledge and practical problem solving.
- For Mixed interviews, include both HR and role-specific technical questions.
- Keep questions clear and suitable for a real interview.
- Do not include answers.
"""
    data = extract_json(call_gemini(prompt))
    questions = data.get("questions", []) if isinstance(data, dict) else data
    clean = []
    for q in questions:
        if isinstance(q, str):
            clean.append({"question": q, "topic": "General"})
        elif isinstance(q, dict) and q.get("question"):
            clean.append({
                "question": str(q["question"]).strip(),
                "topic": str(q.get("topic", "General")).strip(),
            })
    if len(clean) < 1:
        raise ValueError("No questions were returned. Please generate again.")
    return clean[:count]

def evaluate_answer(role, level, interview_type, question, answer):
    prompt = f"""
You are a fair interview coach. Evaluate a candidate's answer for practice only.
Job role: {role}
Candidate level: {level}
Interview type: {interview_type}
Question: {question}
Candidate answer: {answer if answer.strip() else "[No answer provided]"}

Use a fair rubric:
- Accuracy / relevance: 40%
- Completeness: 30%
- Clarity and structure: 20%
- Examples or practical detail: 10%
For HR questions, evaluate relevance, reflection, structure, and concrete examples instead of technical accuracy.
Do not reward length alone. Do not require exact wording. If the answer is empty, give 0.
Return ONLY valid JSON in this format:
{{
  "score": 0,
  "strengths": ["..."],
  "improvements": ["..."],
  "feedback": "A short, kind, specific paragraph.",
  "ideal_answer": "A concise example of a stronger answer."
}}
The score must be an integer from 0 to 100. Do not invent facts about the candidate.
"""
    data = extract_json(call_gemini(prompt))
    score = max(0, min(100, int(float(data.get("score", 0)))))
    strengths = data.get("strengths", [])
    improvements = data.get("improvements", [])
    return {
        "score": score,
        "strengths": strengths if isinstance(strengths, list) else [str(strengths)],
        "improvements": improvements if isinstance(improvements, list) else [str(improvements)],
        "feedback": str(data.get("feedback", "Review the answer and try again.")),
        "ideal_answer": str(data.get("ideal_answer", "No sample answer was returned.")),
    }

def summarize_interview(role, results, overall_score):
    compact = [
        {"question": r["question"], "score": r["score"], "feedback": r["feedback"]}
        for r in results
    ]
    prompt = f"""
Write a short summary for a mock interview.
Role: {role}
Overall score: {overall_score}/100
Question results: {json.dumps(compact, ensure_ascii=False)}
Return 3-5 sentences with strengths, key areas to improve, and one practical next step.
Do not claim this score predicts hiring success.
"""
    return call_gemini(prompt).strip()

# -------------------- UI --------------------
init_db()

st.title("🎯 InterviewAce")
st.subheader("Your AI-powered mock interview coach")
st.write(
    "Choose a job role, practise interview questions, and receive AI-generated feedback. "
    "This is a learning tool; scores are estimates, not hiring decisions."
)

with st.sidebar:
    st.header("Interview setup")
    candidate_name = st.text_input("Your name (optional)")
    role = st.selectbox(
        "Choose job role",
        [
            "Python Developer", "Data Analyst", "Data Scientist",
            "Web Developer", "Software Developer", "AI/ML Engineer",
            "Database Administrator", "HR / General Interview", "Custom Role"
        ],
    )
    custom_role = ""
    if role == "Custom Role":
        custom_role = st.text_input("Enter your target job role")
    actual_role = custom_role.strip() if role == "Custom Role" and custom_role.strip() else role

    level = st.selectbox("Experience level", ["Fresher", "Intermediate", "Experienced"])
    interview_type = st.selectbox("Interview type", ["Technical", "HR", "Mixed"])
    count = st.slider("Number of questions", min_value=3, max_value=10, value=5)
    job_description = st.text_area(
        "Job description (optional)",
        placeholder="Paste a job description to tailor the questions...",
        height=100,
    )
    start_button = st.button("✨ Generate Interview", type="primary", use_container_width=True)

if start_button:
    if role == "Custom Role" and not custom_role.strip():
        st.error("Please enter a custom job role.")
    else:
        with st.spinner("Gemini is preparing your interview questions..."):
            try:
                generated = generate_questions(
                    actual_role, level, interview_type, count, job_description
                )
                st.session_state["interview_config"] = {
                    "candidate_name": candidate_name.strip(),
                    "role": actual_role,
                    "level": level,
                    "interview_type": interview_type,
                    "questions": generated,
                }
                st.session_state["answers"] = {i: "" for i in range(len(generated))}
                st.session_state.pop("evaluation", None)
                st.session_state.pop("saved_interview_id", None)
                st.success(f"Generated {len(generated)} questions. Answer them below.")
            except Exception as exc:
                st.error(f"Could not generate questions: {exc}")

config = st.session_state.get("interview_config")

if config:
    st.divider()
    st.header(f"📝 Mock Interview: {config['role']}")
    st.caption(
        f"{config['level']} · {config['interview_type']} · "
        f"{len(config['questions'])} questions"
    )
    with st.form("answer_form"):
        submitted_answers = {}
        for i, q in enumerate(config["questions"]):
            st.markdown(f"### Question {i + 1}")
            st.write(q["question"])
            st.caption(f"Topic: {q['topic']}")
            submitted_answers[i] = st.text_area(
                "Your answer",
                value=st.session_state.get("answers", {}).get(i, ""),
                key=f"answer_{i}",
                height=120,
                placeholder="Write your answer here...",
            )
        submitted = st.form_submit_button("📊 Evaluate My Answers", type="primary")

    if submitted:
        st.session_state["answers"] = submitted_answers
        results = []
        progress = st.progress(0)
        status = st.empty()
        try:
            for i, q in enumerate(config["questions"]):
                status.write(f"Evaluating answer {i + 1} of {len(config['questions'])}...")
                evaluation = evaluate_answer(
                    config["role"], config["level"], config["interview_type"],
                    q["question"], submitted_answers.get(i, "")
                )
                results.append({
                    "question": q["question"],
                    "topic": q["topic"],
                    "answer": submitted_answers.get(i, ""),
                    **evaluation,
                })
                progress.progress((i + 1) / len(config["questions"]))
            overall_score = round(sum(r["score"] for r in results) / len(results))
            status.write("Creating your interview summary...")
            summary = summarize_interview(config["role"], results, overall_score)
            interview_id = save_interview(
                config["candidate_name"], config["role"], config["level"],
                config["interview_type"], results, overall_score, summary
            )
            st.session_state["evaluation"] = {
                "results": results,
                "overall_score": overall_score,
                "summary": summary,
                "interview_id": interview_id,
            }
            st.session_state["saved_interview_id"] = interview_id
            status.empty()
            st.success("Evaluation complete! Your results have been saved locally.")
        except Exception as exc:
            status.empty()
            st.error(f"Evaluation failed: {exc}")
            st.info("Check your API key, internet connection, model availability, and API quota.")

evaluation = st.session_state.get("evaluation")
if evaluation:
    st.divider()
    st.header("📈 Your Interview Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall score", f"{evaluation['overall_score']}/100")
    col2.metric("Questions answered", f"{len(evaluation['results'])}")
    scores = [r["score"] for r in evaluation["results"]]
    col3.metric("Highest answer score", f"{max(scores)}/100")

    st.subheader("AI coach summary")
    st.write(evaluation["summary"])

    chart_df = pd.DataFrame({
        "Question": [f"Q{i+1}" for i in range(len(evaluation["results"]))],
        "Score": scores,
    }).set_index("Question")
    st.bar_chart(chart_df)

    for i, result in enumerate(evaluation["results"], start=1):
        with st.expander(f"Question {i}: {result['score']}/100 — {result['topic']}"):
            st.markdown("**Question**")
            st.write(result["question"])
            st.markdown("**Your answer**")
            st.write(result["answer"] if result["answer"].strip() else "_No answer provided._")
            st.markdown("**Feedback**")
            st.write(result["feedback"])
            if result["strengths"]:
                st.markdown("**What went well**")
                for item in result["strengths"]:
                    st.write(f"- {item}")
            if result["improvements"]:
                st.markdown("**How to improve**")
                for item in result["improvements"]:
                    st.write(f"- {item}")
            st.markdown("**Example of a stronger answer**")
            st.write(result["ideal_answer"])

    report = {
        "role": config["role"],
        "experience_level": config["level"],
        "interview_type": config["interview_type"],
        "overall_score": evaluation["overall_score"],
        "summary": evaluation["summary"],
        "results": evaluation["results"],
    }
    st.download_button(
        "⬇️ Download interview report (JSON)",
        data=json.dumps(report, indent=2, ensure_ascii=False),
        file_name="interviewace_report.json",
        mime="application/json",
    )

st.divider()
st.header("🗂️ Interview History")
history = load_history()
if history.empty:
    st.info("Your completed interviews will appear here.")
else:
    display_history = history.drop(columns=["id", "feedback"])
    st.dataframe(display_history, use_container_width=True, hide_index=True)
    selected_id = st.selectbox(
        "View a saved interview",
        options=history["id"].tolist(),
        format_func=lambda x: (
            f"#{x} — {history.loc[history['id'] == x, 'role'].iloc[0]} — "
            f"{history.loc[history['id'] == x, 'created_at'].iloc[0]}"
        ),
    )
    saved_answers = load_interview_answers(selected_id)
    if not saved_answers.empty:
        with st.expander("Show saved questions and feedback"):
            st.dataframe(saved_answers, use_container_width=True, hide_index=True)

st.caption(
    "Privacy note: interview answers and results are stored in a local SQLite database "
    "on the machine running this app. Answers are sent to Gemini for evaluation. "
    "Do not enter sensitive personal information."
)
