"""CLI entrypoint for the Deep Research AI Agent.

Tiga mode:

* **Interaktif** (default) — ``python main.py`` tanpa argumen membuka wizard
  singkat: pilihan mode eksekusi → kueri target → konteks tambahan → mode
  kedalaman investigasi → toggle log verbose.
* **Checkpoint CLI** — ``python main.py --query "..." --interactive``: setiap
  siklus berhenti di terminal untuk pivot/expand/selesai + kanvas
  ``graph_interactive.html``.
* **Web App** — ``python main.py --query "..." --web``: server lokal
  (``http://localhost:8000``) menyajikan kanvas mindmap, browser dibuka
  otomatis, dan aksi Pivot/Expand/Stop dikirim via REST (``/api/pivot``,
  ``/api/expand``, ``/api/stop``) sehingga siklus riset berikutnya berjalan
  tanpa mengetik di terminal. Pada TTY (wizard atau ``--interactive``), mode
  ditanyakan di awal: ``[1]`` CLI, ``[2]`` Web App.
* **Non-interaktif** (backward compatible) — ``python main.py --query "..."``
  memakai seluruh flag lama tanpa prompt apa pun, sehingga automated test dan
  skrip tetap berjalan lancar.

Setelah Phase 1 (pencarian + pembersihan) selesai dan laporan ekstraktif
ditampilkan, Phase 2 otomatis berjalan: ``synthesize_llm()`` mengirim segmen
terpilih dari parquet ke provider LLM (Gemini → OpenRouter → Groq) untuk
menghasilkan ``runs/<run_id>/report_llm.md`` + ``runs/<run_id>/report_sidecar.json``
(``verified_facts`` juga dimasukkan ke ``graph.graphml``), yang juga dicetak ke
terminal. Sebelum itu, *agentic critic loop* menilai kecukupan bukti dan
menjalankan sub-kueri gap bila perlu (maks. 2 ronde). Bila tidak ada API key /
library / jaringan, kedua tahap dilewati dengan pesan singkat dan run tetap
sukses; ``--no-llm`` melewatinya secara eksplisit.

Setelah investigasi selesai dan laporan ditampilkan, user dikonfirmasi apakah
folder ``runs/<run_id>`` disimpan; jika tidak, folder dihapus otomatis
(``shutil.rmtree``) supaya tidak menumpuk sampah eksperimen. Prompt ini hanya
muncul bila stdin adalah terminal dan ``--no-prompt`` tidak diberikan — run
non-interaktif tidak pernah menghapus data diam-diam.

Usage:
    python main.py                                   # mode interaktif
    python main.py --query "..."                     # real search + scrape
    python main.py --query "..." --mock-search       # offline, deterministik
    python main.py --query "..." --depth 3           # Deep Dive (min_docs=50, non-strict)
    python main.py --query "..." --no-prompt         # tanpa prompt (CI)
    python main.py --query "..." --no-llm            # lewati sintesis LLM (Phase 2)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

# Make the `drx` package importable without installation.
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drx.brain.llm_synthesizer import format_llm_report, resolve_providers
from drx.config import Settings
from drx.logging_setup import setup_logging
from drx.orchestrator import Orchestrator
from drx.web import WebAppServer

# ---- mode kedalaman investigasi -------------------------------------------
# key -> (label, min_docs, strict_context)
DEPTH_MODES: dict[str, tuple[str, int, bool]] = {
    "1": ("Fast / Dangkal", 10, True),
    "2": ("Balanced / Standar", 25, True),
    "3": ("Deep Dive / Mendalam", 50, False),
}
DEFAULT_DEPTH = "2"
CUSTOM_MIN_DOCS_RANGE = (15, 100)
CUSTOM_MIN_DOCS_DEFAULT = 25


@dataclass
class RunOptions:
    """Resolved run configuration (from the wizard atau CLI flags).

    ``None`` pada ``min_docs``/``strict_context`` berarti "pakai nilai
    settings.yaml" — itu yang menjaga perilaku CLI lama tetap identik.
    """

    query: str
    context: str = ""
    min_docs: int | None = None
    strict_context: bool | None = None
    verbose: bool = True


class _Abort(Exception):
    """User membatalkan wizard (EOF / Ctrl+C)."""


# ---- prompt primitives -----------------------------------------------------

def _stdin_is_tty() -> bool:
    """True hanya bila prompt aman ditampilkan (standing directive: non-TTY)."""
    try:
        return bool(sys.stdin and sys.stdin.isatty())
    except Exception:  # pragma: no cover - closed/detached stdin
        return False


def _prompt_run_mode() -> str:
    """Pilih mode eksekusi di awal: ``cli`` (default) atau ``web`` (Web App).

    Hanya dipanggil pada TTY tanpa ``--no-prompt``; EOF/Ctrl+C aman -> CLI
    (standing directive: run otomatis tidak pernah menggantung menunggu input).
    """
    print()
    print("Pilih mode eksekusi:")
    print("  [1] Mode CLI (terminal interaktif, checkpoint per siklus)")
    print("  [2] Mode Web App (server lokal http://localhost:8000 + browser)")
    try:
        raw = input("Mode [1/2, default 1]: ").strip().lower()
    except (EOFError, KeyboardInterrupt, OSError):
        print()
        return "cli"
    return "web" if raw in {"2", "web", "web app", "webapp"} else "cli"


def _ask(prompt: str) -> str:
    """Read one line; EOF/Ctrl+C abort the wizard instead of looping forever."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt) as exc:
        raise _Abort from exc


def _ask_yes_no(prompt: str, *, default: bool = True) -> bool:
    raw = _ask(prompt).lower()
    if not raw:
        return default
    return raw in {"y", "yes", "ya"}


def _wizard_pick_depth() -> tuple[int, bool]:
    """Menu mode kedalaman (1–4); returns ``(min_docs, strict_context)``."""
    while True:
        raw = _ask(f"   Pilih mode [1/2/3/4] (default {DEFAULT_DEPTH}): ") or DEFAULT_DEPTH
        if raw == "4":
            return _wizard_custom_min_docs(), True
        mode = DEPTH_MODES.get(raw)
        if mode is not None:
            _, min_docs, strict = mode
            return min_docs, strict
        print("   ! Pilihan tidak dikenal — masukkan 1, 2, 3, atau 4.")


def _wizard_custom_min_docs() -> int:
    lo, hi = CUSTOM_MIN_DOCS_RANGE
    while True:
        raw = _ask(
            f"   Masukkan min_docs ({lo}-{hi}) (default {CUSTOM_MIN_DOCS_DEFAULT}): "
        )
        if not raw:
            return CUSTOM_MIN_DOCS_DEFAULT
        try:
            value = int(raw)
        except ValueError:
            print("   ! Harus berupa angka.")
            continue
        if lo <= value <= hi:
            return value
        print(f"   ! Di luar rentang {lo}-{hi}.")


def interactive_setup(
    *,
    default_context: str = "",
    default_verbose: bool = True,
) -> RunOptions:
    """Interactive wizard: target → context → depth mode → verbose toggle."""
    lo, hi = CUSTOM_MIN_DOCS_RANGE
    print()
    print("=" * 64)
    print("  Deep Research AI Agent - Mode Interaktif")
    print("=" * 64)

    query = ""
    while not query:
        query = _ask("1. Kueri Target (nama / entitas): ")
        if not query:
            print("   ! Kueri tidak boleh kosong. Coba lagi.")

    context = _ask(
        "2. Konteks Tambahan "
        '(mis. "Dinas Kesehatan, Kab. Sleman") [kosong = tanpa konteks]: '
    ) or default_context

    print("3. Mode Kedalaman Investigasi:")
    print("     1) Fast / Dangkal       min_docs=10, strict context   (cepat, informasi dasar)")
    print("     2) Balanced / Standar   min_docs=25, strict context   (default, seimbang)")
    print("     3) Deep Dive / Mendalam min_docs=50, tanpa strict     (semua sub-kueri & PDF)")
    print(f"     4) Kustom               min_docs sendiri ({lo}-{hi}), strict context")
    min_docs, strict_context = _wizard_pick_depth()

    verbose = _ask_yes_no(
        f"4. Tampilkan log teknis detail? (Y/n) "
        f"[default: {'Y' if default_verbose else 'n'}]: ",
        default=default_verbose,
    )
    return RunOptions(
        query=query,
        context=context,
        min_docs=min_docs,
        strict_context=strict_context,
        verbose=verbose,
    )


# ---- CLI (backward compatible) --------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep Research AI Agent",
        epilog="Jalankan tanpa argumen (python main.py) untuk mode interaktif.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="research query; bila kosong -> mode interaktif",
    )
    parser.add_argument(
        "--config", default="config/settings.yaml", help="path to settings.yaml"
    )
    parser.add_argument("--run-id", default=None, help="override the run id")
    parser.add_argument("--region", default=None, help="region lock (id, us, jp, global, ...)")
    parser.add_argument(
        "--context",
        "-c",
        default=None,
        help="context terms for 70/30 diversification (e.g. 'Surabaya, Informatika')",
    )
    parser.add_argument("--mock-search", action="store_true", help="offline mock search")
    parser.add_argument(
        "--loose-context",
        action="store_true",
        help="disable strict --context enforcement (keep broad name matches)",
    )
    parser.add_argument(
        "--min-docs",
        type=int,
        default=None,
        help="lazy decomposition threshold: stop after N valid docs (default: settings)",
    )
    parser.add_argument(
        "--depth",
        choices=["1", "2", "3"],
        default=None,
        help="depth preset: 1=Fast (10/strict), 2=Balanced (25/strict), 3=Deep Dive (50/non-strict)",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="jangan tampilkan prompt apa pun (wajib dipakai bersama --query)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="lewati sintesis laporan LLM (Phase 2, report_llm.md)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "checkpoint terminal per siklus (pivot/expand/selesai) + kanvas "
            "graph_interactive.html; hanya aktif bila stdin TTY dan tanpa --no-prompt"
        ),
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help=(
            "Mode Web App: server lokal (default http://localhost:8000), browser "
            "dibuka ke graph_interactive.html; aksi Pivot/Expand/Stop lewat REST"
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="port server Web App (default: settings.web.port = 8000, fallback otomatis)",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="override model LLM sintesis (default: settings.llm.model)",
    )
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def options_from_args(args: argparse.Namespace) -> RunOptions:
    """Build run options from CLI flags; ``None`` = pakai settings.yaml."""
    min_docs = args.min_docs
    # None = "pakai settings.yaml"; --loose-context memaksa non-strict.
    strict: bool | None = False if args.loose_context else None

    if args.depth:
        mode = DEPTH_MODES.get(args.depth)
        if mode is None:  # pragma: no cover - argparse choices guard this
            raise SystemExit(f"--depth harus 1, 2, atau 3 (diberikan: {args.depth!r})")
        _, mode_min_docs, mode_strict = mode
        if min_docs is None:
            min_docs = mode_min_docs
        if not args.loose_context:
            strict = mode_strict

    return RunOptions(
        query=args.query or "",
        context=args.context or "",
        min_docs=min_docs,
        strict_context=strict,
        verbose=args.verbose,
    )


def apply_options(settings: Settings, options: RunOptions) -> None:
    """Apply only explicitly chosen values; ``None`` keeps settings.yaml."""
    if options.min_docs is not None:
        settings.search.lazy_min_docs = options.min_docs
    if options.strict_context is not None:
        settings.cleaning.require_context = options.strict_context


def _llm_provider_note(settings: Settings) -> str:
    """Status sintesis LLM untuk printout konfigurasi (tanpa pernah memanggil API)."""
    if not settings.llm.enabled:
        return "nonaktif"
    env = dict(os.environ)
    for var, value in (
        ("GEMINI_API_KEY", settings.gemini_api_key),
        ("OPENROUTER_API_KEY", settings.openrouter_api_key),
        ("GROQ_API_KEY", settings.groq_api_key),
    ):
        if value:
            env.setdefault(var, value)
    specs = resolve_providers(settings.llm, env)
    if not specs:
        return "nonaktif (tanpa API key)"
    primary = specs[0]
    note = f"{primary.name} / {primary.model}"
    if len(specs) > 1:  # rantai failover antar-provider
        note += " (fallback: " + " -> ".join(s.name for s in specs[1:]) + ")"
    return note


def _print_config(options: RunOptions, *, region: str, llm_note: str = "") -> None:
    strict = (
        "ya (wajib kata kunci konteks)"
        if options.strict_context
        else "tidak (eksplorasi luas)"
    )
    print()
    print("-" * 64)
    print("Konfigurasi investigasi:")
    print(f"  Kueri          : {options.query}")
    print(f"  Konteks        : {options.context or '(tanpa konteks)'}")
    print(f"  min_docs       : {options.min_docs}  (lazy sub-query)")
    print(f"  Strict context : {strict}")
    print(f"  Verbose        : {'ya' if options.verbose else 'tidak'}")
    print(f"  Region         : {region}")
    if llm_note:
        print(f"  Sintesis LLM   : {llm_note}")
    print("-" * 64)
    print()


# ---- run-folder cleanup ----------------------------------------------------

def prompt_save_run(run_dir: Path, *, interactive: bool = True) -> bool:
    """Confirm saving ``runs/<run_id>``; delete the folder when declined.

    Guards (so nothing is ever deleted silently or in automated runs):

    * ``interactive=False`` (``--no-prompt``) -> keep, tanpa bertanya;
    * stdin bukan TTY (pipe/CI/pytest)          -> keep, tanpa bertanya;
    * jawaban kosong/``y``                      -> keep;
    * ``n``                                     -> ``shutil.rmtree`` folder run.
    """
    if not interactive:
        return True
    if not sys.stdin.isatty():
        print(f"(stdin bukan terminal - hasil tetap disimpan di {run_dir})")
        return True
    try:
        keep = _ask_yes_no(
            "\nApakah Anda ingin MENYIMPAN hasil investigasi ini di folder "
            "runs/? (Y/n): ",
            default=True,
        )
    except _Abort:
        print()
        keep = True  # EOF/Ctrl+C: jangan pernah hapus data tanpa jawaban tegas
    if keep:
        print(f"Hasil disimpan di: {run_dir}")
        return True
    shutil.rmtree(run_dir, ignore_errors=True)
    print(f"Folder eksekusi dihapus otomatis: {run_dir}")
    return False


# ---- entrypoint ------------------------------------------------------------

async def amain(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    if args.region:
        settings.search.region = args.region

    # ---- Dual-mode selection (CLI vs Web App) — di awal eksekusi ------------
    # Web App eksplisit (--web) tidak perlu prompt; selebihnya mode ditanyakan
    # hanya pada TTY + bukan --no-prompt (wizard atau --interactive).
    interactive_terminal = not args.no_prompt and _stdin_is_tty()
    mode = "web" if args.web else "cli"
    if (
        not args.web
        and interactive_terminal
        and (args.interactive or args.query is None)
    ):
        mode = _prompt_run_mode()

    if args.query:
        options = options_from_args(args)
    else:
        options = interactive_setup(
            default_context=args.context or "",
            default_verbose=True,
        )
    apply_options(settings, options)
    if args.no_llm:
        settings.llm.enabled = False
    if args.llm_model:
        settings.llm.model = args.llm_model
    setup_logging(options.verbose)

    if args.mock_search:
        # Satu setelan ini mengganti backend untuk AutoSearch utama sekaligus
        # survey pre-flight (keduanya dibangun dari settings.search.backend).
        settings.search.backend = "mock"

    orchestrator = Orchestrator(
        settings,
        run_id=args.run_id,
        context=options.context,
        verbose=options.verbose,
    )

    if args.query is None:
        _print_config(
            options, region=settings.search.region, llm_note=_llm_provider_note(settings)
        )

    # ---- Phase 3: checkpoint interaktif (terminal) / Mode Web App -----------
    use_web = mode == "web" and hasattr(orchestrator, "run_interactive")
    if mode == "web" and not use_web:
        print()
        print("Mode Web App tidak didukung orchestrator ini -> fallback ke CLI.")
    interactive_ok = (
        not args.no_prompt
        and _stdin_is_tty()
        and (args.interactive or args.query is None)
    )
    use_interactive = use_web or (
        interactive_ok and hasattr(orchestrator, "run_interactive")
    )
    if args.interactive and not use_interactive and not use_web:
        print()
        print(
            "Checkpoint interaktif dilewati: stdin bukan TTY, --no-prompt, "
            "atau orchestrator tidak mendukung."
        )

    if use_web:
        web_config = settings.web
        web_server: WebAppServer | None = None
        try:
            web_server = WebAppServer(
                orchestrator,
                host=web_config.host,
                port=args.port if args.port is not None else web_config.port,
                port_attempts=web_config.port_attempts,
                auto_open_browser=web_config.auto_open_browser and not args.no_prompt,
            )
            base_url = web_server.start()
            print()
            print(f"Web App aktif: {base_url}/graph_interactive.html")
            print(
                "Aksi kanvas: Pivot Selected Nodes / Expand Document Pool / "
                "Stop & Generate Final Report -> REST lokal."
            )
            web_server.open_browser()
        except Exception as exc:
            print()
            print(f"Web App gagal start ({exc}) -> fallback ke CLI.")
            web_server = None
        if web_server is not None:
            try:
                report = await orchestrator.run_interactive(
                    options.query,
                    command_source=web_server.next_command,
                    web_api=True,
                )
            finally:
                web_server.stop()
        elif interactive_ok and hasattr(orchestrator, "run_interactive"):
            report = await orchestrator.run_interactive(options.query)
        else:
            report = await orchestrator.run(options.query)
    elif use_interactive:
        report = await orchestrator.run_interactive(options.query)
    else:
        report = await orchestrator.run(options.query)

    md_path = orchestrator.run_dir / "report.md"
    md_path.write_text(report.markdown, encoding="utf-8")
    sidecar_path = orchestrator.run_dir / "report_sidecar.json"
    sidecar_path.write_text(
        json.dumps(report.sidecar, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print()
    print(f"Report written to: {md_path}")
    print("-" * 60)
    print(report.markdown)

    # ---- Phase 2: sintesis laporan final via LLM (otomatis, best-effort) ----
    if args.no_llm:
        print()
        print("LLM synthesis & critic loop: dilewati (--no-llm).")
    else:
        llm_result = await orchestrator.synthesize_llm()
        if llm_result is None:
            print()
            print("LLM synthesis: nonaktif (settings.llm.enabled=false).")
        elif llm_result.ok:
            llm_path = orchestrator.run_dir / "report_llm.md"
            print()
            print(
                f"LLM synthesis ({llm_result.provider}/{llm_result.model}) "
                f"-> {llm_path}"
            )
            if llm_result.sidecar is not None:
                print(
                    "Report sidecar + graph -> "
                    f"{orchestrator.run_dir / 'report_sidecar.json'} "
                    f"({len(llm_result.sidecar.verified_facts)} fakta terverifikasi)"
                )
            print("-" * 60)
            print(format_llm_report(llm_result))
        else:
            reason = llm_result.skipped_reason or llm_result.error or "tidak tersedia"
            print()
            print(f"LLM synthesis dilewati: {reason}")

    prompt_save_run(orchestrator.run_dir, interactive=not args.no_prompt)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.no_prompt and not args.query:
        print(
            "Error: --no-prompt memerlukan --query (mode non-interaktif).",
            file=sys.stderr,
        )
        return 2
    try:
        return asyncio.run(amain(args))
    except _Abort:
        print("\nDibatalkan oleh pengguna.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
