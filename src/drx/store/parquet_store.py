"""Parquet bypass storage — raw cleaned text never re-enters the LLM.

Writes each batch of :class:`Document` to ``runs/<run_id>/parquet/raw_documents.parquet``.
Long text stays on disk here and only short, relevant excerpts are lifted into
the LLM later, which is the core token-saving strategy.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from drx.models import Document

log = logging.getLogger("drx.store.parquet")


class ParquetStore:
    """Append-only document store backed by a single parquet file."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.parquet_dir = self.run_dir / "parquet"
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.parquet_dir / "raw_documents.parquet"

    def save(self, documents: list[Document]) -> None:
        if not documents:
            return
        rows = [d.model_dump() for d in documents]
        df = pd.DataFrame(rows)

        if self.path.exists():
            try:
                old = pd.read_parquet(self.path)
                df = pd.concat([old, df], ignore_index=True)
            except Exception as exc:
                log.warning("could not read existing parquet (%s); rewriting", exc)

        try:
            df.to_parquet(self.path, index=False)
        except Exception as exc:  # graceful fallback so the pipeline never dies
            log.warning("parquet write failed (%s); falling back to JSON", exc)
            self._json_fallback(rows)
        log.debug("stored %d documents -> %s", len(documents), self.path)

    def _json_fallback(self, rows: list[dict]) -> None:
        fallback = self.parquet_dir / "raw_documents.json"
        existing: list[dict] = []
        if fallback.exists():
            try:
                existing = json.loads(fallback.read_text(encoding="utf-8"))
            except Exception:
                existing = []
        existing.extend(rows)
        fallback.write_text(json.dumps(existing, ensure_ascii=False, default=str))

    def count(self) -> int:
        if self.path.exists():
            try:
                return int(pd.read_parquet(self.path).shape[0])
            except Exception:
                pass
        return 0

    def load_documents(self, limit: int = 100) -> list[Document]:
        """Read stored documents back (for synthesis / context packing)."""
        if not self.path.exists():
            return []
        try:
            df = pd.read_parquet(self.path)
        except Exception as exc:
            log.warning("could not read parquet (%s)", exc)
            return []
        if limit and limit > 0:
            df = df.head(limit)
        docs: list[Document] = []
        for record in df.to_dict("records"):
            clean = {k: (None if pd.isna(v) else v) for k, v in record.items()}
            try:
                docs.append(Document(**clean))
            except Exception as exc:
                log.debug("skip malformed document row: %s", exc)
        return docs
