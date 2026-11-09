from __future__ import annotations

from gidr.infrastructure.security.credentials_cli import run


class FakeKeyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.values.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.values[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        del self.values[(service, username)]


def test_set_uses_hidden_prompt_and_never_prints_credential() -> None:
    keyring = FakeKeyring()
    secret = "synthetic-secret-that-must-not-be-printed"
    prompts = iter((secret, secret))
    output: list[str] = []

    result = run(
        ["set", "provider/groq"],
        keyring_loader=lambda: keyring,
        prompt=lambda _: next(prompts),
        output=output.append,
    )

    assert result == 0
    assert keyring.get_password("gidr", "provider/groq") == secret
    assert secret not in "\n".join(output)


def test_set_confirmation_mismatch_does_not_write() -> None:
    keyring = FakeKeyring()
    output: list[str] = []

    result = run(
        ["set", "provider/typesafe"],
        keyring_loader=lambda: keyring,
        prompt=lambda prompt: "synthetic-one" if prompt.startswith("API key") else "synthetic-two",
        output=output.append,
    )

    assert result == 2
    assert keyring.values == {}
    assert all("synthetic-" not in line for line in output)


def test_untrusted_or_missing_backend_fails_closed_before_prompt() -> None:
    output: list[str] = []
    prompted = False

    def prompt(_: str) -> str:
        nonlocal prompted
        prompted = True
        return "synthetic-secret"

    result = run(
        ["set", "provider/gemini"],
        keyring_loader=lambda: None,
        prompt=prompt,
        output=output.append,
    )

    assert result == 2
    assert not prompted
    assert "No credential was read or stored" in output[0]


def test_status_reports_presence_only() -> None:
    keyring = FakeKeyring()
    keyring.set_password("gidr", "provider/siliconflow", "synthetic-status-secret")
    output: list[str] = []

    assert run(["status"], keyring_loader=lambda: keyring, output=output.append) == 0
    assert "provider/siliconflow: configured" in output
    assert "synthetic-status-secret" not in "\n".join(output)
