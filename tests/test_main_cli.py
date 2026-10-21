"""Tests for the interactive CLI wizard, depth presets, and run-folder cleanup.

Covers `main.py` behavior only: the wizard answer flow, the CLI backward
compatibility (flags still work, no prompts in non-interactive runs), the
`--depth` presets, and the final "simpan hasil?" confirmation that can delete
`runs/<run_id>` via `shutil.rmtree`.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import ClassVar

import pytest

import main
from drx.brain.llm_synthesizer import (
    LLMResult,
    ReportSidecar,
    VerifiedFact,
    format_llm_report,
)
from drx.config import Settings
from drx.models import Report

# ---- helpers ---------------------------------------------------------------

def _feed(monkeypatch: pytest.MonkeyPatch, answers: list[str]) -> None:
    """Answer `input()` prompts in order; extra prompts raise EOFError."""

    it = iter(answers)

    def fake_input(prompt: str = "") -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)


class _TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


class _FakeReport(Report):
    def __init__(self) -> None:
        super().__init__(
            run_id="fake",
            query="Sri Lestari",
            markdown="# Fake Report\n",
            sidecar={"docs": 0},
        )


def _fake_orchestrator_factory(
    created: list, *, llm_mode: str = "ok", supports_interactive: bool = True
):
    """Stand-in for `main.Orchestrator` that never touches the network.

    ``llm_mode``: ``"ok"`` writes report_llm.md + returns a result, ``"skip"``
    returns a skipped result (no key), ``"disabled"`` returns ``None``,
    ``"error"`` returns a failed result.
    """

    class _FakeOrchestrator:
        def __init__(self, settings, run_id=None, context="", verbose=False) -> None:
            self.settings = settings
            self.context = context
            self.verbose = verbose
            runs_root = Path(settings.paths.runs_dir)
            self.run_dir = runs_root / (run_id or "fake-run")
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.synthesize_calls = 0
            self.run_calls = 0
            self.interactive_calls = 0
            created.append(self)

        async def run(self, query: str) -> Report:
            self.run_calls += 1
            self.query = query
            return _FakeReport()

        if supports_interactive:

            async def run_interactive(self, query: str, **kwargs) -> Report:
                self.interactive_calls += 1
                self.query = query
                self.interactive_kwargs = kwargs
                return _FakeReport()

        async def synthesize_llm(self):
            self.synthesize_calls += 1
            if llm_mode == "disabled":
                return None
            if llm_mode == "skip":
                return LLMResult(skipped_reason="API key LLM tidak ditemukan")
            if llm_mode == "error":
                return LLMResult(error="RuntimeError: provider down")
            result = LLMResult(
                provider="gemini",
                model="gemini-2.5-flash",
                markdown="# Fake LLM Report\n\n## Ringkasan Eksekutif\nIsi [S1]\n",
                documents_used=1,
                segments_used=1,
                input_tokens_est=10,
                total_tokens=20,
                sections_complete=True,
                sidecar=ReportSidecar(
                    target_entity="Sri Lestari",
                    verified_facts=[
                        VerifiedFact(
                            subject="Sri Lestari",
                            predicate="bekerja sebagai",
                            object="perawat",
                            source_url="https://news.contoh.id/a",
                        )
                    ],
                    sources_used=["https://news.contoh.id/a"],
                ),
                sidecar_complete=True,
            )
            path = self.run_dir / "report_llm.md"
            path.write_text(format_llm_report(result) + "\n", encoding="utf-8")
            (self.run_dir / "report_sidecar.json").write_text(
                json.dumps(result.sidecar.model_dump(), ensure_ascii=False),
                encoding="utf-8",
            )
            return result

    return _FakeOrchestrator


def _write_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "settings.yaml"
    cfg.write_text(f"paths:\n  runs_dir: {tmp_path / 'runs'}\n", encoding="utf-8")
    return cfg


# ---- wizard ----------------------------------------------------------------

def test_wizard_balanced_default(monkeypatch):
    # mode: empty -> default 2; verbose: empty -> default yes
    _feed(monkeypatch, ["Sri Lestari", "Dinas Kesehatan, Kab. Sleman", "", ""])
    opts = main.interactive_setup()
    assert opts.query == "Sri Lestari"
    assert opts.context == "Dinas Kesehatan, Kab. Sleman"
    assert opts.min_docs == 25
    assert opts.strict_context is True
    assert opts.verbose is True


def test_wizard_fast_mode(monkeypatch):
    _feed(monkeypatch, ["Sri Lestari", "", "1", "y"])
    opts = main.interactive_setup()
    assert (opts.min_docs, opts.strict_context, opts.verbose) == (10, True, True)


def test_wizard_deep_dive_mode(monkeypatch):
    _feed(monkeypatch, ["Sri Lestari", "", "3", "n"])
    opts = main.interactive_setup()
    assert (opts.min_docs, opts.strict_context, opts.verbose) == (50, False, False)


def test_wizard_custom_min_docs(monkeypatch):
    _feed(monkeypatch, ["Sri Lestari", "", "4", "77", ""])
    opts = main.interactive_setup()
    assert opts.min_docs == 77
    assert opts.strict_context is True


def test_wizard_custom_min_docs_retries_invalid(monkeypatch):
    _feed(monkeypatch, ["Sri Lestari", "", "4", "abc", "5", "999", "42", ""])
    opts = main.interactive_setup()
    assert opts.min_docs == 42


def test_wizard_unknown_mode_retries(monkeypatch):
    _feed(monkeypatch, ["Sri Lestari", "", "9", "1", ""])
    opts = main.interactive_setup()
    assert opts.min_docs == 10


def test_wizard_empty_query_reprompts(monkeypatch):
    _feed(monkeypatch, ["", "   ", "Sri Lestari", "", "2", ""])
    opts = main.interactive_setup()
    assert opts.query == "Sri Lestari"


def test_wizard_context_default_from_flag(monkeypatch):
    _feed(monkeypatch, ["Sri Lestari", "", "2", ""])
    opts = main.interactive_setup(default_context="Dinas Kesehatan")
    assert opts.context == "Dinas Kesehatan"


def test_wizard_aborts_on_eof(monkeypatch):
    _feed(monkeypatch, [])
    with pytest.raises(main._Abort):
        main.interactive_setup()


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("", True), ("y", True), ("Y", True), ("yes", True), ("ya", True), ("n", False), ("tidak", False)],
)
def test_ask_yes_no(monkeypatch, answer, expected):
    _feed(monkeypatch, [answer])
    assert main._ask_yes_no("simpan?", default=True) is expected


# ---- CLI backward compatibility -------------------------------------------

def test_parse_args_query_optional():
    assert main.parse_args([]).query is None
    assert main.parse_args(["--query", "X"]).query == "X"


def test_options_from_args_without_flags_defers_to_settings():
    opts = main.options_from_args(main.parse_args(["--query", "X"]))
    assert opts.query == "X"
    assert opts.context == ""
    assert opts.min_docs is None  # None -> settings.yaml stays in charge
    assert opts.strict_context is None


def test_options_from_args_depth_presets():
    opts = main.options_from_args(main.parse_args(["--query", "X", "--depth", "1"]))
    assert (opts.min_docs, opts.strict_context) == (10, True)
    opts = main.options_from_args(main.parse_args(["--query", "X", "--depth", "3"]))
    assert (opts.min_docs, opts.strict_context) == (50, False)


def test_options_from_args_explicit_flags_win_over_depth():
    args = main.parse_args(
        ["--query", "X", "--depth", "3", "--min-docs", "12", "--loose-context"]
    )
    opts = main.options_from_args(args)
    assert opts.min_docs == 12
    assert opts.strict_context is False


def test_options_from_args_loose_context_without_depth():
    opts = main.options_from_args(main.parse_args(["--query", "X", "--loose-context"]))
    assert opts.min_docs is None
    assert opts.strict_context is False


def test_parse_args_rejects_unknown_depth():
    with pytest.raises(SystemExit):
        main.parse_args(["--query", "X", "--depth", "9"])


def test_apply_options_respects_none():
    settings = Settings()
    settings.search.lazy_min_docs = 7
    settings.cleaning.require_context = True
    main.apply_options(settings, main.RunOptions(query="x"))
    assert settings.search.lazy_min_docs == 7
    assert settings.cleaning.require_context is True
    main.apply_options(
        settings, main.RunOptions(query="x", min_docs=50, strict_context=False)
    )
    assert settings.search.lazy_min_docs == 50
    assert settings.cleaning.require_context is False


def test_no_prompt_requires_query(capsys):
    assert main.main(["--no-prompt"]) == 2
    assert "--no-prompt" in capsys.readouterr().err


# ---- run-folder cleanup ----------------------------------------------------

def test_prompt_save_run_deletes_when_declined(monkeypatch, tmp_path):
    run_dir = tmp_path / "runs" / "abc"
    (run_dir / "sub").mkdir(parents=True)
    (run_dir / "report.md").write_text("hi", encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", _TTY())
    _feed(monkeypatch, ["n"])
    assert main.prompt_save_run(run_dir, interactive=True) is False
    assert not run_dir.exists()


def test_prompt_save_run_keeps_on_empty_answer(monkeypatch, tmp_path):
    run_dir = tmp_path / "runs" / "abc"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(sys, "stdin", _TTY())
    _feed(monkeypatch, [""])
    assert main.prompt_save_run(run_dir, interactive=True) is True
    assert run_dir.exists()


def test_prompt_save_run_keeps_on_eof(monkeypatch, tmp_path):
    run_dir = tmp_path / "runs" / "abc"
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(sys, "stdin", _TTY())
    _feed(monkeypatch, [])  # EOF at the confirmation prompt
    assert main.prompt_save_run(run_dir, interactive=True) is True
    assert run_dir.exists()


def test_prompt_save_run_skips_prompt_when_no_prompt_flag(monkeypatch, tmp_path):
    run_dir = tmp_path / "runs" / "abc"
    run_dir.mkdir(parents=True)

    def boom(*_args, **_kwargs):
        raise AssertionError("must not prompt under --no-prompt")

    monkeypatch.setattr("builtins.input", boom)
    assert main.prompt_save_run(run_dir, interactive=False) is True
    assert run_dir.exists()


def test_prompt_save_run_skips_prompt_when_stdin_not_tty(monkeypatch, tmp_path):
    run_dir = tmp_path / "runs" / "abc"
    run_dir.mkdir(parents=True)

    def boom(*_args, **_kwargs):
        raise AssertionError("must not prompt without a TTY")

    monkeypatch.setattr("builtins.input", boom)
    monkeypatch.setattr(sys, "stdin", io.StringIO())  # isatty() -> False
    assert main.prompt_save_run(run_dir, interactive=True) is True
    assert run_dir.exists()


# ---- end-to-end entrypoint (orchestrator stubbed, offline) -----------------

def test_main_cli_mode_writes_report_without_prompt(monkeypatch, tmp_path, capsys):
    created: list = []
    monkeypatch.setattr(main, "Orchestrator", _fake_orchestrator_factory(created))

    def boom(*_args, **_kwargs):
        raise AssertionError("CLI mode must not prompt")

    monkeypatch.setattr("builtins.input", boom)
    monkeypatch.setattr(sys, "stdin", io.StringIO())  # not a TTY

    rc = main.main(
        [
            "--query",
            "Sri Lestari",
            "--depth",
            "3",
            "--min-docs",
            "42",
            "--context",
            "Dinas Kesehatan",
            "--config",
            str(_write_config(tmp_path)),
        ]
    )
    assert rc == 0
    orch = created[0]
    assert orch.query == "Sri Lestari"
    assert orch.context == "Dinas Kesehatan"
    assert orch.settings.search.lazy_min_docs == 42  # explicit flag wins
    assert orch.settings.cleaning.require_context is False  # depth 3 -> non-strict
    assert (orch.run_dir / "report.md").exists()
    assert (orch.run_dir / "report_sidecar.json").exists()
    assert "Report written to:" in capsys.readouterr().out


def test_main_interactive_mode_runs_wizard_and_deletes_declined_run(
    monkeypatch, tmp_path, capsys
):
    created: list = []
    monkeypatch.setattr(main, "Orchestrator", _fake_orchestrator_factory(created))
    monkeypatch.setattr(sys, "stdin", _TTY())
    _feed(
        monkeypatch,
        ["1", "Sri Lestari", "Dinas Kesehatan", "3", "n", "n"],  # mode, query, ctx, deep, verbose, save?
    )

    rc = main.main(["--config", str(_write_config(tmp_path))])
    assert rc == 0
    orch = created[0]
    assert orch.settings.search.lazy_min_docs == 50
    assert orch.settings.cleaning.require_context is False
    assert orch.verbose is False
    # '-' in the summary + report printed before the cleanup prompt
    out = capsys.readouterr().out
    assert "Konfigurasi investigasi" in out
    assert "# Fake Report" in out
    assert not orch.run_dir.exists()  # user declined -> shutil.rmtree


def test_main_interactive_mode_keeps_accepted_run(monkeypatch, tmp_path):
    created: list = []
    monkeypatch.setattr(main, "Orchestrator", _fake_orchestrator_factory(created))
    monkeypatch.setattr(sys, "stdin", _TTY())
    _feed(monkeypatch, ["1", "Sri Lestari", "", "2", "", ""])  # mode, save? -> default yes

    assert main.main(["--config", str(_write_config(tmp_path))]) == 0
    assert created[0].run_dir.exists()
    assert (created[0].run_dir / "report.md").exists()


def test_main_mock_search_uses_mock_backend(monkeypatch, tmp_path):
    created: list = []
    monkeypatch.setattr(main, "Orchestrator", _fake_orchestrator_factory(created))
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(sys, "stdin", io.StringIO())

    assert (
        main.main(
            [
                "--query",
                "X",
                "--mock-search",
                "--no-prompt",
                "--config",
                str(_write_config(tmp_path)),
            ]
        )
        == 0
    )
    assert created[0].settings.search.backend == "mock"


# ---- Phase 2: LLM synthesis integration ------------------------------------

def test_main_prints_and_saves_llm_report(monkeypatch, tmp_path, capsys):
    created: list = []
    monkeypatch.setattr(main, "Orchestrator", _fake_orchestrator_factory(created))
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(sys, "stdin", io.StringIO())

    rc = main.main(
        ["--query", "Sri Lestari", "--no-prompt", "--config", str(_write_config(tmp_path))]
    )

    assert rc == 0
    orch = created[0]
    assert orch.synthesize_calls == 1
    llm_path = orch.run_dir / "report_llm.md"
    assert llm_path.exists()
    text = llm_path.read_text(encoding="utf-8")
    assert "# Fake LLM Report" in text
    sidecar_path = orch.run_dir / "report_sidecar.json"
    assert sidecar_path.exists()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["target_entity"] == "Sri Lestari"
    assert sidecar["verified_facts"][0]["predicate"] == "bekerja sebagai"
    out = capsys.readouterr().out
    assert "LLM synthesis (gemini/gemini-2.5-flash)" in out
    assert "Report sidecar + graph ->" in out
    assert "1 fakta terverifikasi" in out
    assert "# Fake LLM Report" in out


def test_main_no_llm_flag_skips_synthesis(monkeypatch, tmp_path, capsys):
    created: list = []
    monkeypatch.setattr(main, "Orchestrator", _fake_orchestrator_factory(created))
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(sys, "stdin", io.StringIO())

    rc = main.main(
        [
            "--query",
            "Sri Lestari",
            "--no-llm",
            "--no-prompt",
            "--config",
            str(_write_config(tmp_path)),
        ]
    )

    assert rc == 0
    orch = created[0]
    assert orch.synthesize_calls == 0  # tidak dipanggil sama sekali
    assert orch.settings.llm.enabled is False
    assert not (orch.run_dir / "report_llm.md").exists()
    assert "LLM synthesis & critic loop: dilewati (--no-llm)." in capsys.readouterr().out


def test_main_llm_skip_note_when_no_api_key(monkeypatch, tmp_path, capsys):
    created: list = []
    monkeypatch.setattr(
        main, "Orchestrator", _fake_orchestrator_factory(created, llm_mode="skip")
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO())

    rc = main.main(
        ["--query", "X", "--no-prompt", "--config", str(_write_config(tmp_path))]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "LLM synthesis dilewati: API key LLM tidak ditemukan" in out
    assert not (created[0].run_dir / "report_llm.md").exists()


def test_main_llm_disabled_note(monkeypatch, tmp_path, capsys):
    created: list = []
    monkeypatch.setattr(
        main, "Orchestrator", _fake_orchestrator_factory(created, llm_mode="disabled")
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO())

    assert (
        main.main(["--query", "X", "--no-prompt", "--config", str(_write_config(tmp_path))])
        == 0
    )
    assert "LLM synthesis: nonaktif" in capsys.readouterr().out


def test_llm_provider_note_reflects_keys(monkeypatch, tmp_path):
    monkeypatch.setattr("drx.config._load_dotenv", lambda *a, **k: None)
    for var in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_GENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "GROQ_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    settings = Settings.load(tmp_path / "missing.yaml")
    settings.gemini_api_key = None
    settings.openrouter_api_key = None
    settings.groq_api_key = None

    assert main._llm_provider_note(settings) == "nonaktif (tanpa API key)"

    settings.gemini_api_key = "gk"
    assert "gemini" in main._llm_provider_note(settings)

    settings.llm.enabled = False
    assert main._llm_provider_note(settings) == "nonaktif"


# ---- Phase 3: interactive checkpoint flag ---------------------------------

def test_interactive_flag_uses_run_interactive_on_tty(monkeypatch, tmp_path):
    created: list = []
    monkeypatch.setattr(main, "Orchestrator", _fake_orchestrator_factory(created))
    monkeypatch.setattr(main, "_stdin_is_tty", lambda: True)
    monkeypatch.setattr(main, "prompt_save_run", lambda *a, **k: True)

    rc = main.main(
        [
            "--query",
            "Sri Lestari",
            "--interactive",
            "--no-llm",
            "--config",
            str(_write_config(tmp_path)),
        ]
    )

    assert rc == 0
    assert created[0].interactive_calls == 1
    assert created[0].run_calls == 0


def test_interactive_flag_falls_back_without_tty(monkeypatch, tmp_path, capsys):
    created: list = []
    monkeypatch.setattr(main, "Orchestrator", _fake_orchestrator_factory(created))
    monkeypatch.setattr(main, "_stdin_is_tty", lambda: False)

    rc = main.main(
        [
            "--query",
            "Sri Lestari",
            "--interactive",
            "--no-llm",
            "--no-prompt",
            "--config",
            str(_write_config(tmp_path)),
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert created[0].run_calls == 1
    assert created[0].interactive_calls == 0
    assert "Checkpoint interaktif dilewati" in out


# ---- Mode Web App (server lokal + endpoint pivot/stop) ---------------------

class _FakeWebServer:
    """Stand-in for main.WebAppServer (tanpa socket, tanpa browser)."""

    instances: ClassVar[list] = []

    def __init__(self, orchestrator, **kwargs) -> None:
        self.orch = orchestrator
        self.kwargs = kwargs
        self.started = False
        self.browser_opened = False
        self.stopped = False
        self.status = "idle"
        _FakeWebServer.instances.append(self)

    def start(self) -> str:
        self.started = True
        self.status = "running"
        return f"http://{self.kwargs.get('host', '127.0.0.1')}:{self.kwargs.get('port', 8000)}"

    def open_browser(self) -> None:
        if self.kwargs.get("auto_open_browser", True):
            self.browser_opened = True

    async def next_command(self) -> dict:
        return {"action": "stop"}

    def stop(self) -> None:
        self.stopped = True
        self.status = "finished"


def test_web_flag_starts_server_and_runs_web_app(monkeypatch, tmp_path, capsys):
    created: list = []
    _FakeWebServer.instances = []
    monkeypatch.setattr(main, "Orchestrator", _fake_orchestrator_factory(created))
    monkeypatch.setattr(main, "WebAppServer", _FakeWebServer)

    rc = main.main(
        [
            "--query",
            "Sri Lestari",
            "--web",
            "--port",
            "8123",
            "--no-llm",
            "--no-prompt",
            "--config",
            str(_write_config(tmp_path)),
        ]
    )

    assert rc == 0
    orch = created[0]
    assert orch.run_calls == 0
    assert orch.interactive_calls == 1
    assert orch.interactive_kwargs.get("web_api") is True
    assert callable(orch.interactive_kwargs.get("command_source"))

    server = _FakeWebServer.instances[0]
    assert server.started and server.stopped
    assert not server.browser_opened  # --no-prompt: jangan buka browser
    assert server.kwargs["port"] == 8123
    out = capsys.readouterr().out
    assert "Web App aktif: http://127.0.0.1:8123/graph_interactive.html" in out


def test_mode_prompt_choice_web_app(monkeypatch, tmp_path):
    created: list = []
    _FakeWebServer.instances = []
    monkeypatch.setattr(main, "Orchestrator", _fake_orchestrator_factory(created))
    monkeypatch.setattr(main, "WebAppServer", _FakeWebServer)
    monkeypatch.setattr(sys, "stdin", _TTY())
    monkeypatch.setattr(main, "prompt_save_run", lambda *a, **k: True)
    _feed(monkeypatch, ["2"])  # mode prompt -> Web App

    rc = main.main(
        [
            "--query",
            "Sri Lestari",
            "--interactive",
            "--no-llm",
            "--config",
            str(_write_config(tmp_path)),
        ]
    )

    assert rc == 0
    assert created[0].interactive_calls == 1
    assert created[0].run_calls == 0
    server = _FakeWebServer.instances[0]
    assert server.started and server.stopped
    assert server.browser_opened  # mode interaktif: browser dibuka otomatis


def test_web_flag_falls_back_without_orchestrator_support(
    monkeypatch, tmp_path, capsys
):
    created: list = []
    monkeypatch.setattr(
        main,
        "Orchestrator",
        _fake_orchestrator_factory(created, supports_interactive=False),
    )
    monkeypatch.setattr(main, "WebAppServer", _FakeWebServer)

    rc = main.main(
        [
            "--query",
            "Sri Lestari",
            "--web",
            "--no-llm",
            "--no-prompt",
            "--config",
            str(_write_config(tmp_path)),
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert created[0].run_calls == 1
    assert created[0].interactive_calls == 0
    assert "Mode Web App tidak didukung" in out
