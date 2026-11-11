"""
SIRIMBA LLM Provider
Centralized provider selection and fallback.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

# Load api/.env without overriding deployment environment variables.
load_dotenv(Path(__file__).resolve().parent / ".env")


def _configured_providers() -> list[str]:
    """Return configured providers in priority order."""
    primary = os.getenv("LLM_PRIMARY_PROVIDER", "gemini").strip().lower()
    fallback = os.getenv("LLM_FALLBACK_PROVIDER", "groq").strip().lower()

    providers = [primary, fallback]

    if os.getenv("LLM_ENABLE_OLLAMA", "false").strip().lower() == "true":
        providers.append("ollama")

    return list(dict.fromkeys(
        provider
        for provider in providers
        if provider in {"gemini", "groq", "ollama"}
    ))


def _create_provider(provider: str):
    """Create a configured LangChain chat model."""
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY belum dikonfigurasi.")

        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            google_api_key=api_key,
            temperature=0,
        )

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY belum dikonfigurasi.")

        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            groq_api_key=api_key,
            temperature=0,
        )

    if provider == "ollama":
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )

    raise ValueError(f"Provider tidak dikenal: {provider}")


def invoke_llm(prompt: str) -> str | None:
    """
    Invoke the first available provider.

    Returns None when all configured providers fail.
    API keys and sensitive exception details are never logged.
    """
    for provider in _configured_providers():
        try:
            model = _create_provider(provider)
            response: BaseMessage = model.invoke(prompt)

            content = response.content
            if isinstance(content, str) and content.strip():
                logger.info("LLM request succeeded using provider: %s", provider)
                return content.strip()

            logger.warning("LLM provider returned empty response: %s", provider)

        except Exception as exc:
            logger.warning(
                "LLM provider %s failed (%s). Trying next provider.",
                provider,
                type(exc).__name__,
            )

    logger.warning("No configured LLM provider is currently available.")
    return None
