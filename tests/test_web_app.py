"""Tests for Mode Web App: stdlib local server + async command bridge.

Covers ``/api/state``, ``/api/graph``, HTML serving/placeholder, and the three
control endpoints (``/api/pivot``, ``/api/expand``, ``/api/stop``) including
node-ID validation, stop-wins semantics, and port fallback.
"""
from __future__ import annotations

import asyncio
import json
import socket
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from drx.models import Fact, SourceRef
from drx.store.graph_store import GraphStore
from drx.web.server import WebAppServer


class _FakeOrch:
    """Orchestrator stand-in: state + graph store, tanpa pipeline riset."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_id = "web-run"
        self.main_query = "Sri Lestari"
        self.entity = "Sri Lestari"
        self.context = "Sleman"
        self.cycle = 1
        self.tokens_total = 1234
        self.critic_suggestions = [
            {"node_id": "n1", "label": "Sri Lestari", "reason": "target utama"},
            {"node_id": "n99", "label": "Hilang", "reason": "tidak ada di graf"},
        ]
        self.settings = SimpleNamespace(visualizer=SimpleNamespace(max_nodes=50))
        self.interactive = True
        self._pivot_history = {"n1"}
        self.budget = SimpleNamespace(
            status=lambda: {
                "tokens": 1234,
                "max_tokens": 150000,
                "tokens_remaining": 148766,
                "urls": 12,
                "max_urls": 200,
                "urls_remaining": 188,
                "elapsed_seconds": 42.5,
                "max_runtime_seconds": 600,
                "seconds_remaining": 557.5,
            }
        )
        self.graph_store = GraphStore(run_dir)
        self.graph_store.resolve_and_add(
            [
                Fact(
                    subject="Sri Lestari",
                    predicate="bekerja_di",
                    object="Dinas Kesehatan Sleman",
                    provenance=[
                        SourceRef(
                            url="https://a.id",
                            source_id="S1",
                            quote="Sri Lestari bekerja di Dinas Kesehatan Sleman",
                        )
                    ],
                )
            ]
        )

    def estimated_cost_usd(self) -> float:
        return 0.0004


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _get_json(url: str) -> tuple[int, dict]:
    status, raw = _get(url)
    return status, json.loads(raw.decode("utf-8"))


def _post_json(url: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload or {}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _start(tmp_path: Path, **kwargs) -> tuple[WebAppServer, _FakeOrch]:
    orch = _FakeOrch(tmp_path / "runs" / "web-run")
    orch.run_dir.mkdir(parents=True, exist_ok=True)
    server = WebAppServer(orch, port=0, auto_open_browser=False, **kwargs)
    server.start()
    return server, orch


@pytest.mark.asyncio
async def test_state_graph_and_html_endpoints(tmp_path: Path) -> None:
    server, orch = _start(tmp_path)
    try:
        status, state = await asyncio.to_thread(_get_json, f"{server.url}/api/state")
        assert status == 200
        assert state["run_id"] == "web-run"
        assert state["cycle"] == 1
        assert state["tokens"] == 1234
        assert state["cost_usd"] == pytest.approx(0.0004)
        assert state["nodes"] >= 2 and state["edges"] >= 1
        assert state["status"] in {"running", "awaiting"}
        assert state["html_ready"] is False
        assert state["budget"]["tokens_remaining"] == 148766

        status, payload = await asyncio.to_thread(_get_json, f"{server.url}/api/graph")
        assert status == 200
        assert payload["meta"]["api"] is True
        assert payload["meta"]["budget"]["max_urls"] == 200
        assert payload["meta"]["pivoted_nodes"] == ["n1"]
        node_ids = {node["id"] for node in payload["vis"]["nodes"]}
        assert node_ids  # vis.js-ready nodes
        assert payload["vis"]["edges"][0]["id"] == "e0"
        # saran tanpa node di graf (n99) difilter di payload kanvas
        assert [s["node_id"] for s in payload["suggestions"]] == ["n1"]

        # placeholder sebelum siklus pertama menulis kanvas
        status, raw = await asyncio.to_thread(_get, f"{server.url}/graph_interactive.html")
        assert status == 200
        assert "Menunggu siklus pertama" in raw.decode("utf-8")

        # kanvas asli disajikan tanpa cache setelah orchestrator merender
        canvas = orch.run_dir / "graph_interactive.html"
        canvas.write_text("<html><body>KANVAS ASLI</body></html>", encoding="utf-8")
        status, raw = await asyncio.to_thread(_get, f"{server.url}/graph_interactive.html")
        assert status == 200
        assert "KANVAS ASLI" in raw.decode("utf-8")

        status, _ = await asyncio.to_thread(_get_json, f"{server.url}/api/hilang")
        assert status == 404
    finally:
        server.stop()


@pytest.mark.asyncio
async def test_pivot_endpoint_validates_nodes_and_enqueues(tmp_path: Path) -> None:
    server, _orch = _start(tmp_path)
    try:
        status, body = await asyncio.to_thread(
            _post_json, f"{server.url}/api/pivot", {"node_ids": []}
        )
        assert status == 400 and body["ok"] is False

        status, body = await asyncio.to_thread(
            _post_json, f"{server.url}/api/pivot", {"node_ids": ["n42"]}
        )
        assert status == 400
        assert "n42" in body["error"]

        status, body = await asyncio.to_thread(
            _post_json, f"{server.url}/api/pivot", {"node_ids": ["n1"]}
        )
        assert status == 200 and body["ok"] is True
        command = await server.next_command()
        assert command == {"action": "pivot", "node_ids": ["n1"]}
    finally:
        server.stop()


@pytest.mark.asyncio
async def test_expand_endpoint_enqueues_command(tmp_path: Path) -> None:
    server, _orch = _start(tmp_path)
    try:
        status, body = await asyncio.to_thread(
            _post_json, f"{server.url}/api/expand"
        )
        assert status == 200 and body["ok"] is True
        assert await server.next_command() == {"action": "expand"}
    finally:
        server.stop()


@pytest.mark.asyncio
async def test_stop_endpoint_wins_over_pending_pivot(tmp_path: Path) -> None:
    server, _orch = _start(tmp_path)
    try:
        await asyncio.to_thread(_post_json, f"{server.url}/api/pivot", {"node_ids": ["n1"]})
        await asyncio.to_thread(_post_json, f"{server.url}/api/pivot", {"node_ids": ["n2"]})
        status, body = await asyncio.to_thread(_post_json, f"{server.url}/api/stop")
        assert status == 200 and body["ok"] is True

        # Stop membuang seluruh perintah pivot yang tertunda.
        assert await server.next_command() == {"action": "stop"}
        assert server.stop_requested is True

        # Setelah stop, aksi pivot/expand baru ditolak.
        status, body = await asyncio.to_thread(
            _post_json, f"{server.url}/api/pivot", {"node_ids": ["n1"]}
        )
        assert status == 409 and body["ok"] is False
    finally:
        server.stop()


@pytest.mark.asyncio
async def test_stop_wakes_pending_next_command(tmp_path: Path) -> None:
    server, _orch = _start(tmp_path)
    try:
        waiting = asyncio.create_task(server.next_command())
        await asyncio.sleep(0.05)
        assert server.status == "awaiting"

        await asyncio.to_thread(_post_json, f"{server.url}/api/stop")
        command = await asyncio.wait_for(waiting, timeout=5)
        assert command == {"action": "stop"}
    finally:
        server.stop()


@pytest.mark.asyncio
async def test_next_command_returns_stop_after_finished(tmp_path: Path) -> None:
    server, _orch = _start(tmp_path)
    try:
        server.mark_finished()
        assert await server.next_command() == {"action": "stop"}
        assert server.status == "finished"
    finally:
        server.stop()


@pytest.mark.asyncio
async def test_start_falls_back_when_port_is_taken(tmp_path: Path) -> None:
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    occupied = blocker.getsockname()[1]

    orch = _FakeOrch(tmp_path / "runs" / "web-run")
    orch.run_dir.mkdir(parents=True, exist_ok=True)
    server = WebAppServer(
        orch,
        port=occupied,
        port_attempts=5,
        auto_open_browser=False,
    )
    try:
        url = server.start()
        assert server.port != occupied
        assert f":{server.port}" in url
        status, state = await asyncio.to_thread(_get_json, f"{server.url}/api/state")
        assert status == 200 and state["status"] in {"running", "awaiting"}
    finally:
        server.stop()
        blocker.close()


def test_start_without_event_loop_raises(tmp_path: Path) -> None:
    """``start()`` butuh running loop (dipanggil dari ``asyncio.run`` main)."""
    orch = _FakeOrch(tmp_path / "runs" / "web-run")
    server = WebAppServer(orch, port=0, auto_open_browser=False)
    with pytest.raises(RuntimeError):
        server.start()
