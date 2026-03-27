"""
rag_pipeline.py — Retrieval-Augmented Generation Pipeline
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

VECTORSTORE_DIR = Path("vectorstore")
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K_RETRIEVAL = 4
MEMORY_WINDOW = 10
TEMPERATURE = 0.2

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """You are a precise and reliable customer support AI.

Answer ONLY using the provided context.

Rules:
- If not in context, say: "I don't have that information in my knowledge base."
- Do NOT hallucinate
- Be concise
- Use bullet points when helpful

Context:
{context}
"""

CONDENSE_QUESTION_TEMPLATE = """Given chat history and a follow-up question,
rewrite it as a standalone question.

Chat History:
{chat_history}

Question:
{question}

Standalone question:"""

# ── Data Classes ──────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ChatResponse:
    answer: str
    sources: list[dict] = field(default_factory=list)
    error: Optional[str] = None

# ── Vector Store ──────────────────────────────────────────────────────────────

def load_vectorstore(vectorstore_dir: Path = VECTORSTORE_DIR) -> FAISS:
    index_file = vectorstore_dir / "index.faiss"

    if not index_file.exists():
        raise FileNotFoundError("Run ingest.py first.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing OPENAI_API_KEY")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=api_key,
    )

    return FAISS.load_local(
        str(vectorstore_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )

# ── Build Chain ───────────────────────────────────────────────────────────────

def build_chain(vectorstore: FAISS) -> ConversationalRetrievalChain:
    api_key = os.getenv("OPENAI_API_KEY")

    retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

    memory = ConversationBufferWindowMemory(
    k=MEMORY_WINDOW,
    memory_key="chat_history",
    output_key="answer",
    return_messages=True,
)

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        openai_api_key=api_key,
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
        HumanMessagePromptTemplate.from_template("{question}"),
    ])

    condense_prompt = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        condense_question_prompt=condense_prompt,
        return_source_documents=True,
    )

# ── Source Formatter ──────────────────────────────────────────────────────────

def _format_sources(docs: list) -> list[dict]:
    seen = set()
    results = []

    for doc in docs:
        meta = doc.metadata or {}
        source = meta.get("source", "Unknown")
        page = meta.get("page")
        score = meta.get("score")

        snippet = doc.page_content[:200].replace("\n", " ").strip()

        key = (source, page)
        if key in seen:
            continue
        seen.add(key)

        results.append({
            "file": Path(source).name if source != "Unknown" else "Unknown",
            "page": page,
            "snippet": snippet + ("..." if len(doc.page_content) > 200 else ""),
            "score": round(score, 3) if isinstance(score, (int, float)) else None,
        })

    return results
# ── Chatbot ───────────────────────────────────────────────────────────────────

class RAGChatbot:

    def __init__(self, vectorstore_dir: Path = VECTORSTORE_DIR):
        vs = load_vectorstore(vectorstore_dir)
        self._chain = build_chain(vs)

    def chat(self, question: str) -> ChatResponse:
        if not question.strip():
            return ChatResponse(answer="Please enter a question.", sources=[])

        try:
            query = " ".join(question.lower().strip().split())
            result = self._chain.invoke({"question": query})
            answer = result.get("answer", "")
            docs = result.get("source_documents", [])
            sources = _format_sources(docs)
            return ChatResponse(answer=answer, sources=sources)

        except Exception as e:
            logger.exception("Chat pipeline failed")
            return ChatResponse(
                answer="Error processing request.",
                sources=[],
                error=str(e),
            )

    def clear_memory(self):
        self._chain.memory.clear()