import time
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "https://docchat-ai-b1mw.onrender.com").rstrip("/")
REQUEST_TIMEOUT_SEC = int(os.getenv("REQUEST_TIMEOUT_SEC", "60"))
RETRY_COUNT = int(os.getenv("RETRY_COUNT", "2"))
QUESTION_STARTERS = {
    "who", "what", "when", "where", "why", "how", "which", "whose",
    "is", "are", "can", "could", "would", "should", "do", "does", "did",
}
QA_INTENT_HINTS = {
    "tell", "explain", "describe", "list", "name", "give", "show", "summarize",
}
TOXICITY_HINTS = {
    "kill", "hurt", "hate", "idiot", "stupid", "trash", "loser",
    "threat", "abuse", "shut up", "useless",
}

st.set_page_config(page_title="DocChat AI", page_icon="AI", layout="wide")

st.markdown(
    """
    <style>
    .stChatMessage { border-radius: 14px; padding: 8px; }
    .safe-label { color: #2fbf71; font-weight: 700; }
    .risk-label { color: #ff5b5b; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "document" not in st.session_state:
    st.session_state.document = ""
if "inference_in_progress" not in st.session_state:
    st.session_state.inference_in_progress = False


def call_api(method: str, path: str, payload: dict | None = None):
    url = f"{API_URL}{path}"
    last_error = None
    for attempt in range(RETRY_COUNT + 1):
        try:
            response = requests.request(
                method=method,
                url=url,
                json=payload,
                timeout=REQUEST_TIMEOUT_SEC,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt < RETRY_COUNT:
                time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"Request failed after retries: {last_error}")


def infer_auto_mode(prompt: str, has_document: bool) -> str:
    """Return 'qa' or 'toxicity' for Auto mode."""
    text = prompt.strip().lower()
    if not has_document:
        return "toxicity"
    if "?" in text:
        return "qa"

    first_word = text.split()[0] if text.split() else ""
    if first_word in QUESTION_STARTERS:
        return "qa"
    if first_word in QA_INTENT_HINTS:
        return "qa"
    if any(hint in text for hint in TOXICITY_HINTS):
        return "toxicity"
    # In document-enabled auto mode, default to Q&A for neutral text.
    return "qa"


with st.sidebar:
    st.title("Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    mode = st.selectbox("Mode", ["Auto", "Document Q&A", "Toxicity Detection"])
    if mode == "Auto":
        st.caption("Auto mode sends question-like text to Q&A and other text to toxicity.")
    st.divider()

    try:
        health = call_api("GET", "/health")
        if health.get("model_loaded"):
            adapter_state = "Adapter Active" if health.get("adapter_loaded") else "Base Model Only"
            st.success(f"Server Online ({adapter_state})")
        else:
            st.warning(f"Server degraded: {health.get('error', 'Model unavailable')}")
    except Exception as exc:
        st.error(f"Server Offline: {exc}")

    st.subheader("Document Context")
    uploaded = st.file_uploader("Upload .txt or .md", type=["txt", "md"])
    if uploaded is not None:
        content = uploaded.read().decode("utf-8", errors="ignore")
        if content.strip():
            st.session_state.document = content
            st.success("Loaded document from file.")
        else:
            st.warning("Uploaded file is empty.")

    doc_input = st.text_area("Paste document here...", value=st.session_state.document, height=300)
    if doc_input != st.session_state.document:
        st.session_state.document = doc_input

st.title("DocChat AI")
st.caption("Fine-tuned local LLM for document Q&A and toxicity detection")

if st.session_state.inference_in_progress:
    st.info("A previous request was interrupted by refresh. You can safely resend your message.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

prompt = st.chat_input("Ask a question about the document or classify comment toxicity...")
if prompt is not None:
    prompt = prompt.strip()
    if not prompt:
        st.warning("Please enter a non-empty message.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                st.session_state.inference_in_progress = True
                try:
                    if mode == "Document Q&A":
                        selected_mode = "qa"
                    elif mode == "Toxicity Detection":
                        selected_mode = "toxicity"
                    else:
                        selected_mode = infer_auto_mode(prompt, bool(st.session_state.document.strip()))

                    use_qa = selected_mode == "qa"
                    if use_qa:
                        if not st.session_state.document.strip():
                            answer = "Document is empty. Paste or upload a document first."
                        else:
                            data = call_api(
                                "POST",
                                "/ask",
                                {"document": st.session_state.document, "question": prompt},
                            )
                            answer = data.get("answer", "No response from model.")
                    else:
                        data = call_api("POST", "/predict", {"comment": prompt})
                        label = data.get("label", "unknown").upper()
                        confidence = float(data.get("confidence", 0.0))
                        explanation = data.get("explanation", "No explanation provided.")
                        if label == "UNKNOWN":
                            answer = (
                                f"**Label:** <span class='risk-label'>{label}</span>\n\n"
                                f"**Confidence:** {confidence:.2f}\n\n"
                                f"**Reason:** {explanation}"
                            )
                        elif confidence < conf_threshold:
                            answer = "Response filtered due to low confidence threshold."
                        else:
                            css_class = "safe-label" if label == "SAFE" else "risk-label"
                            answer = (
                                f"**Label:** <span class='{css_class}'>{label}</span>\n\n"
                                f"**Confidence:** {confidence:.2f}\n\n"
                                f"**Reason:** {explanation}"
                            )

                    st.markdown(answer, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as exc:
                    error_msg = f"Error connecting to API: {exc}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                finally:
                    st.session_state.inference_in_progress = False

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()
