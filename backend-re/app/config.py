"""Configuration for the LangGraph backend."""

import os
from functools import lru_cache
from typing import List

from dotenv import load_dotenv
from pydantic_settings import BaseSettings


load_dotenv()


class Settings(BaseSettings):
    """Application settings."""

    app_name: str = "LangGraph智能旅行助手"
    app_version: str = "2.0.0"
    debug: bool = False

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = (
        "http://localhost:5173,http://localhost:3000,"
        "http://127.0.0.1:5173,http://127.0.0.1:3000"
    )

    amap_api_key: str = ""

    llm_model_id: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_timeout: int = 60
    llm_temperature: float = 0.3

    log_level: str = "INFO"

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",
    }

    def get_cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


def validate_config() -> None:
    settings = get_settings()
    errors = []
    warnings = []

    if not settings.amap_api_key:
        errors.append("AMAP_API_KEY未配置")

    api_key = settings.llm_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        warnings.append("LLM_API_KEY或OPENAI_API_KEY未配置, LLM功能可能无法使用")

    if errors:
        raise ValueError("配置错误:\n" + "\n".join(f"  - {error}" for error in errors))

    if warnings:
        print("\n⚠️  配置警告:")
        for warning in warnings:
            print(f"  - {warning}")


def print_config() -> None:
    settings = get_settings()
    print(f"应用名称: {settings.app_name}")
    print(f"版本: {settings.app_version}")
    print(f"服务器: {settings.host}:{settings.port}")
    print(f"高德地图API Key: {'已配置' if settings.amap_api_key else '未配置'}")
    print(f"LLM API Key: {'已配置' if (settings.llm_api_key or os.getenv('OPENAI_API_KEY')) else '未配置'}")
    print(f"LLM Base URL: {settings.llm_base_url}")
    print(f"LLM Model: {settings.llm_model_id}")
    print(f"日志级别: {settings.log_level}")
