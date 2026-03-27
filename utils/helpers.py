from typing import Optional
"""
utils/helpers.py — Shared Utility Functions
--------------------------------------------
Small helpers used by both app.py and rag_pipeline.py.
Kept here so neither module grows with unrelated concerns.
"""

import os
from pathlib import Path


def check_env() -> tuple[bool, str]:
    """
    Verify OPENAI_API_KEY is present and non-empty.
    Returns (ok: bool, message: str).
    """
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return False, (
            "**OPENAI_API_KEY** is not set.\n\n"
            "Add it to your `.env` file:\n```\nOPENAI_API_KEY=sk-...\n```"
        )
    if not key.startswith("sk-"):
        return False, (
            "**OPENAI_API_KEY** looks invalid (should start with `sk-`).\n"
            "Please check your `.env` file."
        )
    return True, "API key detected ✓"


def check_vectorstore(vectorstore_dir: Path = Path("vectorstore")) -> tuple[bool, str]:
    """
    Check whether a FAISS index has been built.
    Returns (exists: bool, message: str).
    """
    index = vectorstore_dir / "index.faiss"
    if index.exists():
        return True, f"Vector store found at `{vectorstore_dir}/` ✓"
    return False, (
        f"No vector store found at `{vectorstore_dir}/`.\n\n"
        "**Run the ingestion pipeline first:**\n```bash\npython ingest.py\n```"
    )


from typing import Optional

def format_source_badge(file: str, page: Optional[int]) -> str:
    """
    Build a compact citation label, e.g. 'faq.pdf · p.3' or 'manual.txt'.
    """
    label = file
    if page is not None:
        label += f" · p.{page + 1}"   # FAISS pages are 0-indexed
    return label
