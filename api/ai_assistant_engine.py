"""
SIRIMBA AI Assistant Engine

Komponen:
- Ollama local LLM
- LangChain analytics tools
- Dependency injection untuk fungsi backend aktif

Modul ini tidak memuat dataset sendiri.
Tidak mengubah AI Prediction, MAI, atau report generator.
"""

from typing import Callable, Optional

from langchain_ollama import ChatOllama
from langchain.tools import tool


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_MODEL = "qwen2.5:1.5b"
OLLAMA_BASE_URL = "http://localhost:11434"


# ============================================================
# BACKEND ANALYTICS INJECTION
# ============================================================

_query_growth_function: Optional[Callable] = None
_compare_growth_function: Optional[Callable] = None


def set_analytics_functions(
    query_function: Callable,
    compare_function: Callable,
) -> None:
    """
    Menghubungkan tools dengan fungsi analitik backend aktif.

    Fungsi backend tetap menggunakan dataset global yang sudah dimuat.
    """

    global _query_growth_function
    global _compare_growth_function

    _query_growth_function = query_function
    _compare_growth_function = compare_function


# ============================================================
# LLM INITIALIZATION
# ============================================================

def create_llm() -> ChatOllama:
    """Membuat instance Qwen melalui Ollama lokal."""

    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )


# ============================================================
# BASIC LLM INVOKE
# ============================================================

def ask_llm(question: str) -> str:
    """Mengirim pertanyaan ke Qwen melalui Ollama."""

    llm = create_llm()
    response = llm.invoke(question)

    return response.content


# ============================================================
# LANGCHAIN ANALYTICS TOOLS
# ============================================================

@tool
def query_growth_tool(question: str) -> str:
    """
    Mengambil statistik pertumbuhan pohon dari dataset SIRIMBA.
    Gunakan untuk pertanyaan faktual tentang nilai pertumbuhan,
    wilayah, kompartemen, spesies, klon, atau periode.
    """

    if _query_growth_function is None:
        return (
            "Fungsi analitik backend belum terhubung. "
            "Tools belum siap digunakan."
        )

    result = _query_growth_function(question)

    return str(result)


@tool
def compare_growth_tool(question: str) -> str:
    """
    Membandingkan statistik pertumbuhan pohon antar kelompok.
    Gunakan untuk pertanyaan perbandingan wilayah,
    kompartemen, spesies, atau klon.
    """

    if _compare_growth_function is None:
        return (
            "Fungsi analitik backend belum terhubung. "
            "Tools belum siap digunakan."
        )

    result = _compare_growth_function(question)

    if result is None:
        return (
            "Tidak ditemukan hasil perbandingan. "
            "Pertanyaan mungkin belum dikenali sebagai perbandingan."
        )

    return str(result)