"""Mode Web App — server lokal (stdlib) untuk kanvas mindmap + REST kontrol.

Alur:

1. ``main.py`` membuat :class:`WebAppServer` di atas orchestrator yang sama.
2. ``start()`` menyalakan ``ThreadingHTTPServer`` di ``127.0.0.1`` (port 8000,
   fallback 8001..8009) pada thread daemon, lalu browser dibuka ke
   ``/graph_interactive.html``.
3. Halaman kanvas (``web_api=True``) mem-POST aksi user:
   ``/api/pivot`` (batch node ID), ``/api/expand``, ``/api/stop`` — handler
   meneruskan perintah ke ``asyncio.Queue`` milik loop utama lewat
   ``call_soon_threadsafe``, sehingga siklus riset berikutnya berjalan di
   orchestrator tanpa memblokir server.
4. Halaman polling ``/api/graph`` tiap 2 detik dan meng-update vis.js DataSet
   (node/edge) + panel saran + statistik token/biaya secara real-time.

Tidak ada dependency baru: ``http.server`` + ``asyncio`` sudah cukup. Server
hanya listen di loopback dan tidak pernah menulis file; render kanvas tetap
dilakukan orchestrator (``runs/<run_id>/graph_interactive.html``).
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from drx.visualizer.graph_view import build_payload, vis_elements

log = logging.getLogger("drx.web")

_PLACEHOLDER_HTML = """<!DOCTYPE html>
<html lang="id"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="2">
<title>DRX Web App - menunggu siklus pertama</title>
<style>
  html, body { background: #F8FAFC; color: #0F172A; font-family: Inter, system-ui, sans-serif;
    background-image: radial-gradient(#CBD5E1 1px, transparent 1px); background-size: 20px 20px; }
  .card { max-width: 520px; margin: 18vh auto; background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 12px; padding: 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.08); }
  .muted { color: #64748B; font-size: 13px; }
  code { font-family: "Fira Code", monospace; color: #0E7490; }
</style></head>
<body>
  <div class="card">
    <h2 style="margin:0 0 8px">Menunggu siklus pertama&hellip;</h2>
    <p class="muted">Server Web App berjalan, tetapi kanvas belum dirender. Halaman ini
    refresh otomatis tiap 2 detik.</p>
    <p class="muted">Metrik: <code>/api/state</code> &middot; kanvas: <code>/api/graph</code></p>
  </div>
</body></html>
"""


class WebAppServer:
    """Server HTTP loopback + jembatan perintah async untuk Mode Web App."""

    def __init__(
        self,
        orchestrator: Any,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        port_attempts: int = 10,
        auto_open_browser: bool = True,
        html_name: str = "graph_interactive.html",
    ) -> None:
        self.orch = orchestrator
        self.host = host
        self.port = int(port)
        self.port_attempts = max(1, int(port_attempts))
        self.auto_open_browser = bool(auto_open_browser)
        self.html_name = html_name

        self.httpd: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._commands: asyncio.Queue[dict] | None = None
        self.status = "idle"  # idle | awaiting | running | finished
        self.stop_requested = False
        self._closed = False

    # ---- lifecycle ---------------------------------------------------------

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> str:
        """Bind + serve di thread daemon; kembalikan base URL (mis. :8000)."""
        self._loop = asyncio.get_running_loop()
        self._commands = asyncio.Queue()
        handler = self._make_handler()
        last_exc: Exception | None = None
        for offset in range(self.port_attempts):
            candidate = 0 if self.port == 0 else self.port + offset
            try:
                self.httpd = ThreadingHTTPServer((self.host, candidate), handler)
                break
            except OSError as exc:  # port dipakai -> coba port berikutnya
                last_exc = exc
                continue
        if self.httpd is None:
            raise RuntimeError(
                f"tidak bisa bind Web App di {self.host}:{self.port} "
                f"(+{self.port_attempts - 1} port fallback): {last_exc}"
            )
        self.port = int(self.httpd.server_address[1])
        self.status = "running"
        self.thread = threading.Thread(
            target=self.httpd.serve_forever,
            kwargs={"poll_interval": 0.2},
            name="drx-web",
            daemon=True,
        )
        self.thread.start()
        log.info("Web App server: %s/graph_interactive.html", self.url)
        return self.url

    def open_browser(self) -> None:
        """Buka browser ke kanvas (best-effort, tidak pernah raise)."""
        if not self.auto_open_browser:
            return
        target = f"{self.url}/{self.html_name}"
        try:
            threading.Thread(
                target=webbrowser.open,
                args=(target,),
                name="drx-browser",
                daemon=True,
            ).start()
        except Exception as exc:  # pragma: no cover - browser opsional
            log.debug("gagal membuka browser: %s", exc)

    def stop(self) -> None:
        """Matikan server + bangunkan ``next_command`` yang sedang menunggu."""
        self._closed = True
        self.status = "finished"
        self._wake_stop()
        httpd, self.httpd = self.httpd, None
        if httpd is not None:
            try:
                httpd.shutdown()
            except Exception:  # pragma: no cover - best effort
                pass
            try:
                httpd.server_close()
            except Exception:  # pragma: no cover
                pass
        thread, self.thread = self.thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def mark_finished(self) -> None:
        """Tandai run selesai (tanpa mematikan server; halaman terakhir tetap bisa dibuka)."""
        self.status = "finished"

    # ---- command bridge ------------------------------------------------------

    async def next_command(self) -> dict:
        """Tunggu perintah berikutnya dari browser; ``stop`` bila server tutup."""
        if self._closed or self.status == "finished" or self._commands is None:
            return {"action": "stop"}
        self.status = "awaiting"
        try:
            command = await self._commands.get()
        finally:
            if not self._closed:
                self.status = "running"
        if not isinstance(command, dict):
            return {"action": "stop"}
        return command

    def _enqueue(self, command: dict) -> None:
        if self._loop is None or self._commands is None or self._closed:
            raise RuntimeError("Web App server belum berjalan")
        self._loop.call_soon_threadsafe(self._commands.put_nowait, command)

    def _enqueue_stop(self) -> None:
        """Stop selalu menang: buang perintah tertunda, lalu antrekan stop."""
        self.stop_requested = True
        if self._commands is not None:
            while True:
                try:
                    self._commands.get_nowait()
                except asyncio.QueueEmpty:
                    break
            self._commands.put_nowait({"action": "stop"})

    def _request_stop(self) -> None:
        if self._loop is None:
            raise RuntimeError("Web App server belum berjalan")
        self._loop.call_soon_threadsafe(self._enqueue_stop)

    def _wake_stop(self) -> None:
        if self._loop is None or self._commands is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._enqueue_stop)
        except RuntimeError:  # loop sudah ditutup
            pass

    # ---- state / payload ------------------------------------------------------

    def _run_dir(self) -> Path | None:
        run_dir = getattr(self.orch, "run_dir", None)
        return Path(run_dir) if run_dir else None

    def _graph_store(self) -> Any:
        return getattr(self.orch, "graph_store", None)

    def _cost_usd(self) -> float:
        cost_fn = getattr(self.orch, "estimated_cost_usd", None)
        try:
            return float(cost_fn()) if callable(cost_fn) else 0.0
        except Exception:
            return 0.0

    def _target(self) -> str:
        entity = str(getattr(self.orch, "entity", "") or "")
        return entity or str(getattr(self.orch, "main_query", "") or "")

    def _budget_status(self) -> dict:
        budget = getattr(self.orch, "budget", None)
        data: dict = {}
        try:
            if budget is not None and callable(getattr(budget, "status", None)):
                data = dict(budget.status())
            elif budget is not None and callable(getattr(budget, "snapshot", None)):
                data = dict(budget.snapshot())
        except Exception:
            data = {}
        overrides = getattr(self.orch, "_budget_overrides", set()) or set()
        try:
            data["overrides"] = sorted(str(item) for item in overrides)
        except Exception:
            data["overrides"] = []
        data["interactive"] = bool(getattr(self.orch, "interactive", False))
        return data

    def _pivot_history(self) -> list[str]:
        try:
            return sorted(
                str(item)
                for item in (getattr(self.orch, "_pivot_history", set()) or set())
            )
        except Exception:
            return []

    def _max_nodes(self) -> int:
        settings = getattr(self.orch, "settings", None)
        visualizer = getattr(settings, "visualizer", None)
        return int(getattr(visualizer, "max_nodes", 600) or 600)

    def snapshot(self) -> dict:
        """Ringkasan status run untuk ``/api/state``."""
        store = self._graph_store()
        try:
            nodes = len(store.node_records()) if store is not None else 0
            edges = len(store.edge_records()) if store is not None else 0
        except Exception:
            nodes = edges = 0
        run_dir = self._run_dir()
        html_ready = bool(run_dir and (run_dir / self.html_name).exists())
        return {
            "run_id": str(getattr(self.orch, "run_id", "") or ""),
            "target": self._target(),
            "cycle": int(getattr(self.orch, "cycle", 0) or 0),
            "tokens": int(getattr(self.orch, "tokens_total", 0) or 0),
            "cost_usd": self._cost_usd(),
            "budget": self._budget_status(),
            "status": self.status,
            "suggestions": list(getattr(self.orch, "critic_suggestions", []) or []),
            "nodes": nodes,
            "edges": edges,
            "html_ready": html_ready,
            "stop_requested": self.stop_requested,
        }

    def graph_payload(self) -> dict:
        """Payload kanvas terkini (node/edge/suggestion + vis.js elements)."""
        payload = build_payload(
            self._graph_store(),
            target_entity=self._target(),
            context=str(getattr(self.orch, "context", "") or ""),
            critic_suggestions=list(getattr(self.orch, "critic_suggestions", []) or []),
            cycle=int(getattr(self.orch, "cycle", 0) or 0),
            tokens=int(getattr(self.orch, "tokens_total", 0) or 0),
            cost_usd=self._cost_usd(),
            max_nodes=self._max_nodes(),
            run_id=str(getattr(self.orch, "run_id", "") or ""),
            budget=self._budget_status(),
            pivoted_nodes=self._pivot_history(),
        )
        payload["meta"]["status"] = self.status
        payload["meta"]["api"] = True
        payload["meta"]["interactive"] = bool(getattr(self.orch, "interactive", False))
        payload["vis"] = vis_elements(payload)
        return payload

    def _html_bytes(self) -> bytes:
        run_dir = self._run_dir()
        path = run_dir / self.html_name if run_dir else None
        if path is not None and path.exists():
            try:
                return path.read_bytes()
            except Exception as exc:  # pragma: no cover - file sedang ditulis
                log.debug("gagal baca kanvas %s: %s", path, exc)
        return _PLACEHOLDER_HTML.encode("utf-8")

    # ---- HTTP handler ---------------------------------------------------------

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        app = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "DRXWeb/1.0"

            def log_message(self, format: str, *args: Any) -> None:
                log.debug("web %s - %s", self.address_string(), format % args)

            # -- helpers --
            def _send(
                self,
                status: int,
                body: bytes,
                content_type: str,
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):  # pragma: no cover
                    pass

            def _json(self, status: int, payload: dict) -> None:
                body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
                self._send(status, body, "application/json; charset=utf-8")

            def _read_json(self) -> dict:
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    return {}
                if length <= 0 or length > 1_000_000:
                    return {}
                try:
                    data = json.loads(self.rfile.read(length).decode("utf-8"))
                except Exception:
                    return {}
                return data if isinstance(data, dict) else {}

            # -- routes --
            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path in ("/", f"/{app.html_name}"):
                    self._send(200, app._html_bytes(), "text/html; charset=utf-8")
                    return
                if path == "/api/state":
                    self._json(200, app.snapshot())
                    return
                if path == "/api/graph":
                    try:
                        self._json(200, app.graph_payload())
                    except Exception as exc:
                        log.warning("payload graf gagal: %s", exc)
                        self._json(
                            503,
                            {"ok": False, "error": f"graf belum siap: {exc}"},
                        )
                    return
                if path == "/favicon.ico":
                    self._send(204, b"", "image/x-icon")
                    return
                self._json(404, {"ok": False, "error": "rute tidak ditemukan"})

            def do_POST(self) -> None:
                path = urlparse(self.path).path
                if path not in ("/api/pivot", "/api/expand", "/api/stop"):
                    self._json(404, {"ok": False, "error": "rute tidak ditemukan"})
                    return
                if app._closed:
                    self._json(409, {"ok": False, "error": "server sudah ditutup"})
                    return
                if app.stop_requested and path != "/api/stop":
                    self._json(
                        409,
                        {"ok": False, "error": "run sudah dihentikan; tidak ada aksi baru"},
                    )
                    return
                if path == "/api/pivot":
                    body = self._read_json()
                    raw_ids = body.get("node_ids")
                    if not isinstance(raw_ids, list) or not raw_ids:
                        self._json(
                            400,
                            {
                                "ok": False,
                                "error": "body wajib {\"node_ids\": [\"n1\", ...]}",
                            },
                        )
                        return
                    node_ids = [str(item).strip() for item in raw_ids if str(item).strip()]
                    store = app._graph_store()
                    graph = getattr(store, "graph", None) if store is not None else None
                    if graph is None:
                        self._json(
                            409,
                            {"ok": False, "error": "graf belum siap (siklus pertama belum selesai)"},
                        )
                        return
                    invalid = [node_id for node_id in node_ids if node_id not in graph]
                    if invalid:
                        self._json(
                            400,
                            {
                                "ok": False,
                                "error": "ID node tidak ditemukan: " + ", ".join(invalid),
                            },
                        )
                        return
                    try:
                        app._enqueue({"action": "pivot", "node_ids": node_ids})
                    except RuntimeError as exc:
                        self._json(409, {"ok": False, "error": str(exc)})
                        return
                    self._json(
                        200,
                        {
                            "ok": True,
                            "message": "Pivot dikirim: " + ", ".join(node_ids),
                            "node_ids": node_ids,
                        },
                    )
                    return
                if path == "/api/expand":
                    try:
                        app._enqueue({"action": "expand"})
                    except RuntimeError as exc:
                        self._json(409, {"ok": False, "error": str(exc)})
                        return
                    self._json(
                        200,
                        {
                            "ok": True,
                            "message": (
                                "Expand dikirim: mencari gelombang dokumen "
                                "level-1 baru (deep search + sudut baru)"
                            ),
                        },
                    )
                    return
                # /api/stop
                try:
                    app._request_stop()
                except RuntimeError as exc:
                    self._json(409, {"ok": False, "error": str(exc)})
                    return
                self._json(
                    200,
                    {
                        "ok": True,
                        "message": "Stop dikirim: menyusun laporan final",
                    },
                )

        return Handler


__all__ = ["WebAppServer"]
