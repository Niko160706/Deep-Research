"""Vector store stub.

Phase 2 replaces the metadata-only backing with an INT8 BGE-small embedding
worker + a FAISS flat index persisted to ``vectors.faiss``. Phase 1 keeps
argument cards as JSON metadata so the full loop remains runnable.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from drx.models import EvidenceCard

log = logging.getLogger("drx.store.vector")


class VectorStore:
    """Card metadata store (FAISS index arrives in Phase 2)."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.meta_path = self.run_dir / "vectors_metadata.json"
        self._cards: list[dict] = self._load()

    def add(self, cards: list[EvidenceCard]) -> None:
        self._cards.extend(c.model_dump() for c in cards)
        self._flush()

    def card_count(self) -> int:
        return len(self._cards)

    def top_cards(self, k: int = 10) -> list[dict]:
        return self._cards[:k]

    # ---- persistence ----
    def _flush(self) -> None:
        self.meta_path.write_text(
            json.dumps(self._cards, ensure_ascii=False, default=str), encoding="utf-8"
        )

    def _load(self) -> list[dict]:
        if self.meta_path.exists():
            try:
                return json.loads(self.meta_path.read_text(encoding="utf-8"))
            except Exception:
                log.warning("could not read %s; starting empty", self.meta_path)
        return []
