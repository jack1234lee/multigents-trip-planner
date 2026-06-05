"""LangChain chat model factory."""

import os

from langchain_openai import ChatOpenAI

from ..config import get_settings


_llm_instance: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    """Return a singleton OpenAI-compatible LangChain chat model."""

    global _llm_instance

    if _llm_instance is None:
        settings = get_settings()
        api_key = settings.llm_api_key or os.getenv("OPENAI_API_KEY")
        base_url = settings.llm_base_url or os.getenv("OPENAI_BASE_URL")

        _llm_instance = ChatOpenAI(
            model=settings.llm_model_id,
            api_key=api_key,
            base_url=base_url,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout,
        )

        print("✅ LangChain LLM服务初始化成功")
        print(f"   模型: {settings.llm_model_id}")
        print(f"   Base URL: {base_url}")

    return _llm_instance


def reset_llm() -> None:
    global _llm_instance
    _llm_instance = None
