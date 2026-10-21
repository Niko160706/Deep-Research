"""Unit tests for configuration and .env loading."""
from __future__ import annotations

import os

from drx.config import Settings, _load_dotenv


def test_load_dotenv_sets_missing_vars(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("BRAVE_SEARCH_API_KEY=test_key_123\n# a comment\n\n", encoding="utf-8")
    _load_dotenv(env_file)
    assert os.environ["BRAVE_SEARCH_API_KEY"] == "test_key_123"


def test_load_dotenv_does_not_override_existing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "existing_key")
    env_file = tmp_path / ".env"
    env_file.write_text("BRAVE_SEARCH_API_KEY=new_key\n", encoding="utf-8")
    _load_dotenv(env_file)
    assert os.environ["BRAVE_SEARCH_API_KEY"] == "existing_key"


def test_settings_reads_brave_key_from_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "env_key")
    settings = Settings.load(tmp_path / "missing.yaml")
    assert settings.brave_search_api_key == "env_key"


def test_settings_reads_tavily_key_from_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-env-key")
    settings = Settings.load(tmp_path / "missing.yaml")
    assert settings.tavily_api_key == "tvly-env-key"


def test_llm_config_defaults() -> None:
    settings = Settings()
    assert settings.llm.enabled is True
    assert settings.llm.provider == "auto"
    assert settings.llm.model == "gemini-3.8-flash"
    assert settings.llm.fallback_model == "gemini-3.7-flash,gemini-3.5-flash"
    assert settings.llm.openrouter_model == "google/gemini-3.8-flash"
    assert settings.llm.groq_model == "qwen/qwen3.8-27b"
    assert (
        settings.llm.groq_fallback_model
        == "openai/gpt-oss-120b,llama-3.3-70b-versatile"
    )
    assert settings.llm.provider_min_interval_seconds == 2.0  # Groq OTPM guard
    assert settings.llm.thinking_level == "low"
    assert settings.llm.max_segments == 18
    assert settings.llm.max_input_tokens == 30000
    # agentic critic loop (fact verifier sebelum sintesis)
    assert settings.llm.critic_enabled is True
    assert settings.llm.critic_max_loops == 2
    assert settings.llm.critic_max_sub_queries == 3
    assert settings.llm.critic_max_segments == 8
    assert settings.llm.critic_max_input_tokens == 3400
    # Groq OTPM free tier 1000 -> output critic/ekstraksi dibatasi 350-400.
    assert settings.llm.critic_max_output_tokens == 400
    assert settings.llm.extraction_max_output_tokens == 400


def test_web_config_defaults() -> None:
    settings = Settings()
    assert settings.web.enabled is True
    assert settings.web.host == "127.0.0.1"
    assert settings.web.port == 8000
    assert settings.web.auto_open_browser is True
    assert settings.web.port_attempts == 10


def test_settings_reads_gemini_key_from_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gem-env-key")
    settings = Settings.load(tmp_path / "missing.yaml")
    assert settings.gemini_api_key == "gem-env-key"


def test_settings_reads_google_api_key_alias(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("drx.config._load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-alias-key")
    settings = Settings.load(tmp_path / "missing.yaml")
    assert settings.gemini_api_key == "google-alias-key"
