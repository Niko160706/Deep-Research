"""Security/config validation uses synthetic canaries, never real credentials."""

import json

import pytest

from gidr.bootstrap import configure
from gidr.config import SettingsInputError, load_settings


@pytest.mark.parametrize(
    "data",
    [
        {"runtime": {"active_run_limit": 2}},
        {"runtime": {"workers": 0}},
        {"network": {"max_response_bytes": -1}},
        {"model": {"sdk_retries": 2}},
        {"source": {"github_clone_default": True}},
        {"source": {"firecrawl_enabled": True}},
        {"parser": {"allow_source_code_execution": True}},
        {"security": {"plaintext_keyring_fallback": True}},
        {"model": {"decision_primary": "other-provider"}},
        {"runtime": {"workers": "3"}},
        {"output": {"locale": "unsupported"}},
        {"audit": {"include_raw_credentials": True}},
    ],
)
def test_invalid_or_unsafe_config_rejected(data):
    with pytest.raises(SettingsInputError):
        load_settings(data)


@pytest.mark.parametrize(
    "data",
    [
        {"api_key": "SYNTHETIC-CREDENTIAL-CANARY"},
        {"model": {"api_key": "SYNTHETIC-CREDENTIAL-CANARY"}},
        {"model": {"approved_credential_aliases": ["SYNTHETIC-CREDENTIAL-CANARY"]}},
        {"model": {"approved_credential_aliases": ("SYNTHETIC-CREDENTIAL-CANARY",)}},
        {"output": {"locale": "SYNTHETIC-CREDENTIAL-CANARY"}},
    ],
)
def test_config_error_never_echoes_secret(data):
    with pytest.raises(SettingsInputError) as caught:
        load_settings(data)
    assert "SYNTHETIC-CREDENTIAL-CANARY" not in str(caught.value)
    assert "SYNTHETIC-CREDENTIAL-CANARY" not in repr(caught.value)
    assert caught.value.__cause__ is None


def test_settings_do_not_load_dotenv_or_process_secrets(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("TYPESAFE_API_KEY=SYNTHETIC-CREDENTIAL-CANARY\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TYPESAFE_API_KEY", "SYNTHETIC-CREDENTIAL-CANARY")
    monkeypatch.setenv("GIDR_RUNTIME__WORKERS", "99")
    settings = configure()
    assert settings.runtime.workers == 3
    assert settings.network.enabled is False
    assert settings.security.dotenv_mode == "DISABLED"
    assert settings.model.approved_credential_aliases == ()
    assert "SYNTHETIC-CREDENTIAL-CANARY" not in repr(settings)
    assert "SYNTHETIC-CREDENTIAL-CANARY" not in json.dumps(settings.public_snapshot())


def test_aliases_are_not_values_and_snapshot_is_detached():
    settings = load_settings({"model": {"approved_credential_aliases": ("provider/typesafe",)}})
    snapshot = settings.public_snapshot()
    assert snapshot["model"]["approved_credential_aliases"] == ["provider/typesafe"]
    snapshot["runtime"]["workers"] = 100
    assert settings.runtime.workers == 3
