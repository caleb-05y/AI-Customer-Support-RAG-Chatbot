from typing import Optional
import streamlit as st
from pathlib import Path
from rag_pipeline import RAGChatbot, ChatResponse

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 RAG Chatbot")

# ── Initialize Chatbot ────────────────────────────────────────────────────────
VECTORSTORE_DIR = Path("vectorstore")

@st.cache_resource(show_spinner=False)
def get_chatbot():
    return RAGChatbot(vectorstore_dir=VECTORSTORE_DIR)

chatbot = get_chatbot()

# ── Session State for Multi-turn Chat ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── User Input ────────────────────────────────────────────────────────────────
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Enter your question:", "")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    with st.spinner("Chatbot is thinking..."):
        response: ChatResponse = chatbot.chat(user_input)

        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({
            "role": "bot",
            "content": response.answer,
            "error": response.error,
        })

        if response.sources:
            st.session_state.messages.append({
                "role": "sources",
                "content": "\n".join(
                    f"- {s['file']} (page {s.get('page', '?')}): {s['snippet']}"
                    for s in response.sources
                )
            })
# ── Display Conversation ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    elif msg["role"] == "bot":
        st.markdown(f"**Bot:** {msg['content']}")
        if msg.get("error"):
            st.error(msg["error"])
    elif msg["role"] == "sources":
        st.markdown(f"**Sources:**\n{msg['content']}")