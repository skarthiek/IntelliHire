import streamlit as st
import os
from dotenv import load_dotenv
from streamlit_lottie import st_lottie
import requests

load_dotenv()

import google.generativeai as genai

gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

from pdf_parser import extract_text_from_pdf
from csv_parser import extract_text_from_csv
from rag_pipeline import create_vectorstore, get_answer

def get_hr_advice(resume_text, job_desc_text):
    prompt = f"""
You are an experienced HR professional. Given the following resume and job description, provide detailed, human-like advice on how the candidate can improve their resume to better fit the job. Be constructive and specific.

Resume:
{resume_text}

Job Description:
{job_desc_text}

Advice:
"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    advice = response.text.strip()
    return advice

def compute_ats_score(resume_text, job_desc_text):
    resume_words = set(resume_text.lower().split())
    job_desc_words = set(job_desc_text.lower().split())
    common_words = resume_words.intersection(job_desc_words)
    score = len(common_words) / len(job_desc_words) if job_desc_words else 0
    return round(score * 100, 2)

def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

st.set_page_config(page_title="RAG Chatbot", layout="wide")

lottie_animation = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")

user_type = st.sidebar.selectbox("Are you a staff or a student?", ["Select", "Staff", "Student"])

if user_type == "Student":
    st.title("ATS Resume Analyzer")

    resume_file = st.file_uploader("Upload your Resume PDF file", type=["pdf"])
    job_desc_text = st.text_area("Paste the Job Description text here")

    if st.button("Analyze ATS Score and Get HR Advice"):
        if not resume_file or not job_desc_text:
            st.error("Please upload your Resume PDF and provide the Job Description text.")
        else:
            with st.spinner("Extracting text from resume..."):
                with open("temp_resume.pdf", "wb") as f:
                    f.write(resume_file.getbuffer())
                resume_text = extract_text_from_pdf("temp_resume.pdf")
            score = compute_ats_score(resume_text, job_desc_text)
            st.success(f"ATS Score: {score}%")
            with st.spinner("Generating HR advice..."):
                advice = get_hr_advice(resume_text, job_desc_text)
            st.markdown("### HR Advice to Improve Your Resume")
            st.write(advice)

elif user_type == "Staff":
    st.title("Staff Panel")

    st_lottie(lottie_animation, height=200)

    st.sidebar.title("Upload your files (PDF or CSV)")
    uploaded_files = st.sidebar.file_uploader("Choose files", type=["pdf", "csv"], accept_multiple_files=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None

    if uploaded_files:
        with st.spinner("Processing files..."):
            all_text = ""
            for uploaded_file in uploaded_files:
                file_path = f"data/{uploaded_file.name}"
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                if uploaded_file.name.endswith(".pdf"):
                    text = extract_text_from_pdf(file_path)
                elif uploaded_file.name.endswith(".csv"):
                    text = extract_text_from_csv(file_path)
                else:
                    st.error(f"Unsupported file format: {uploaded_file.name}")
                    continue

                all_text += text + "\n"

            st.session_state.vector_db = create_vectorstore(all_text)
            st.success("processed successfully!")

    st.divider()
    st.subheader("Ask a question")

    query = st.text_input("Type your question here...", placeholder="e.g. What is the summary of these documents?")

    if st.button("Ask") and query:
        if st.session_state.vector_db is None:
            st.error("Please upload and process files first.")
        else:
            with st.spinner("Thinking..."):
                answer = get_answer(st.session_state.vector_db, query)
                st.session_state.chat_history.append(("You", query))
                st.session_state.chat_history.append(("Chatbot", answer))

    if st.session_state.chat_history:
        for sender, message in reversed(st.session_state.chat_history):
            with st.chat_message(name=sender):
                st.markdown(message)

else:
    st.title("Welcome to the RAG Chatbot Application")
    st.write("Please select if you are a Staff or a Student from the sidebar to proceed.")
