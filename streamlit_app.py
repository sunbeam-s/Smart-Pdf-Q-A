import asyncio
from pathlib import Path
import time

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests
import base64

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

st.set_page_config(page_title="PDF Q&A", page_icon="📄", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    h1 {
        color: #ffffff !important;
        text-align: center;
        font-weight: 800 !important;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.25);
        animation: fadeInDown 0.8s ease-out;
    }

    h2, h3, .stMarkdown p {
        color: #ffffff !important;
    }

    [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] {
        color: #ffffff !important;
        font-weight: 600;
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .block-container {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2.5rem !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }

    div.stButton > button, div.stFormSubmitButton > button {
        background: linear-gradient(90deg, #ff6a88 0%, #c471ed 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.8rem;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.25s ease-in-out;
        box-shadow: 0 4px 14px rgba(196, 113, 237, 0.4);
    }

    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        transform: scale(1.06);
        box-shadow: 0 6px 20px rgba(196, 113, 237, 0.6);
        color: white;
    }

    div.stButton > button:active, div.stFormSubmitButton > button:active {
        transform: scale(0.97);
    }

    .stTextInput input, .stNumberInput input {
        border-radius: 10px !important;
        border: 2px solid #ffffff !important;
        color: #000000 !important;
        background-color: #ffffff !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        border-radius: 14px;
        border: 2px dashed #ffffff !important;
        background: rgba(255,255,255,0.06);
    }

    .answer-card {
        background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%);
        color: white;
        padding: 1.3rem 1.6rem;
        border-radius: 16px;
        margin-top: 0.8rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        animation: popIn 0.4s ease-out;
    }

    .source-chip {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        color: #fff;
        padding: 0.25rem 0.8rem;
        border-radius: 999px;
        margin: 0.2rem 0.3rem 0.2rem 0;
        font-size: 0.85rem;
        border: 1px solid rgba(255,255,255,0.3);
    }

    @keyframes popIn {
        0% { opacity: 0; transform: scale(0.9); }
        100% { opacity: 1; transform: scale(1); }
    }

    hr, [data-testid="stDivider"] {
        border-color: rgba(255,255,255,0.25) !important;
    }

    .khuj-loader {
        display: flex;
        align-items: center;
        gap: 14px;
        background: rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 1rem 1.4rem;
        margin: 0.8rem 0;
        animation: popIn 0.3s ease-out;
    }

    .khuj-spinner {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        border: 4px solid rgba(255,255,255,0.25);
        border-top: 4px solid #ffffff;
        animation: spin 0.8s linear infinite;
    }

    .khuj-text {
        color: #ffffff;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.3px;
    }

    .khuj-dots span {
        animation: blink 1.4s infinite;
        opacity: 0;
    }
    .khuj-dots span:nth-child(1) { animation-delay: 0s; }
    .khuj-dots span:nth-child(2) { animation-delay: 0.2s; }
    .khuj-dots span:nth-child(3) { animation-delay: 0.4s; }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    @keyframes blink {
        0%, 100% { opacity: 0; }
        50% { opacity: 1; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def khuj_loader(text="Khujtechi"):
    return st.markdown(
        f"""
        <div class="khuj-loader">
            <div class="khuj-spinner"></div>
            <div class="khuj-text">
                {text}
                <span class="khuj-dots"><span>.</span><span>.</span><span>.</span></span>
                &nbsp;ektu wait koren please! 🔍
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    event_key = os.getenv("INNGEST_EVENT_KEY")
    return inngest.Inngest(
        app_id="rag_app",
        is_production=bool(event_key),
        event_key=event_key,
    )


async def send_rag_ingest_event(file) -> None:
    client = get_inngest_client()
    file_bytes = file.getbuffer()
    encoded = base64.b64encode(file_bytes).decode("utf-8")
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_base64": encoded,
                "source_id": file.name,
            },
        )
    )


async def send_rag_query_event(question: str, top_k: int) -> str:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )
    return result[0]


def _inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "https://api.inngest.com/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    headers = {"Authorization": f"Bearer {os.getenv('INNGEST_SIGNING_KEY')}"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)


st.title("📄PDF Q&A")

st.header("1. PDF ta Upload Koren")
uploaded = st.file_uploader("Ekta PDF chose koren", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    if st.button("🚀 Ingest Kore Den!"):
        loader_slot = st.empty()
        with loader_slot.container():
            khuj_loader("Upload hocche")
        asyncio.run(send_rag_ingest_event(uploaded))
        loader_slot.empty()
        st.success(f"Hoye Gèche!: {uploaded.name} 🎉")

st.divider()

st.header("2. ki jante chacchen?!")
with st.form("rag_query_form"):
    question = st.text_input("ki jante chacchen?!")
    top_k = st.number_input("Koto ta chunk khujbo? (optional)", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("✨ Here is the answer!")

if submitted and question.strip():
    loader_slot = st.empty()
    with loader_slot.container():
        khuj_loader("Khujtechi")
    try:
        event_id = asyncio.run(send_rag_query_event(question.strip(), int(top_k)))
        output = wait_for_run_output(event_id)
        answer = output.get("answer", "")
        sources = output.get("sources", [])

        loader_slot.empty()

        st.subheader("Ei je apnr Answer")
        st.markdown(
            f'<div class="answer-card">{answer or "(Kisu pai nai re bhai)"}</div>',
            unsafe_allow_html=True,
        )

        if sources:
            st.caption("Kothay theke pailam")
            chips = "".join(f'<span class="source-chip">{s}</span>' for s in sources)
            st.markdown(chips, unsafe_allow_html=True)
    except Exception as e:
        loader_slot.empty()
        st.error(f"kisu ekta gondogol hoise: {e}")