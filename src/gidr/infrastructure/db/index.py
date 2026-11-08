"""Bounded lexical queries; external-content updates share the commit boundary."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from gidr.contracts.common import ChunkId, Contract, SourceId
from gidr.infrastructure.audit.events import event_for
from gidr.infrastructure.db.authority import PersistenceAuthority, WriteContext


class IndexEntry(Contract):
    source_id: SourceId
    chunk_id: ChunkId | None = None
    title: Annotated[str, Field(max_length=4096)] = ""
    snippet: Annotated[str, Field(max_length=8192)] = ""
    extracted_text: Annotated[str, Field(max_length=262144)] = ""


class IndexResult(Contract):
    count: int


class RebuildRequest(Contract):
    action: Literal["REBUILD"] = "REBUILD"


def remove_rows(context: WriteContext, source_id: str, chunk_id: str | None) -> None:
    rows = context.execute(
        "SELECT rowid,title,snippet,extracted_text FROM source_index WHERE source_id=? AND chunk_key=?",
        (source_id, chunk_id or ""),
    ).fetchall()
    for row in rows:
        context.execute(
            "INSERT INTO source_fts(source_fts,rowid,title,snippet,extracted_text) VALUES('delete',?,?,?,?)",
            (row["rowid"], row["title"], row["snippet"], row["extracted_text"]),
        )
        context.execute("DELETE FROM source_index WHERE rowid=?", (row["rowid"],))


class LexicalIndex:
    def __init__(self, authority: PersistenceAuthority):
        self.authority = authority

    def upsert(self, command_id: str, trace_id: str, entry: IndexEntry) -> IndexResult:
        self.authority.policy.validate_text(entry.title + entry.snippet + entry.extracted_text)

        def change(context: WriteContext) -> IndexResult:
            remove_rows(context, entry.source_id, entry.chunk_id)
            cursor = context.execute(
                "INSERT INTO source_index(source_id,chunk_id,title,snippet,extracted_text) VALUES(?,?,?,?,?)",
                (entry.source_id, entry.chunk_id, entry.title, entry.snippet, entry.extracted_text),
            )
            context.execute(
                "INSERT INTO source_fts(rowid,title,snippet,extracted_text) VALUES(?,?,?,?)",
                (cursor.lastrowid, entry.title, entry.snippet, entry.extracted_text),
            )
            context.emit(
                event_for(
                    context,
                    component="indexing",
                    event_type="index.updated",
                    aggregate_id=entry.source_id,
                )
            )
            return IndexResult(count=1)

        return self.authority.execute(command_id, trace_id, entry, IndexResult, change)

    def remove(self, command_id: str, trace_id: str, entry: IndexEntry) -> IndexResult:
        def change(context: WriteContext) -> IndexResult:
            remove_rows(context, entry.source_id, entry.chunk_id)
            context.emit(
                event_for(
                    context,
                    component="indexing",
                    event_type="index.removed",
                    aggregate_id=entry.source_id,
                )
            )
            return IndexResult(count=0)

        return self.authority.execute(command_id, trace_id, entry, IndexResult, change)

    def rebuild(self, command_id: str, trace_id: str) -> IndexResult:
        def change(context: WriteContext) -> IndexResult:
            context.execute("INSERT INTO source_fts(source_fts) VALUES('rebuild')")
            context.execute("INSERT INTO source_fts(source_fts,rank) VALUES('integrity-check',1)")
            context.emit(
                event_for(
                    context,
                    component="indexing",
                    event_type="index.rebuilt",
                    aggregate_id="INDEX-source",
                )
            )
            return IndexResult(
                count=context.execute("SELECT count(*) FROM source_index").fetchone()[0]
            )

        return self.authority.execute(command_id, trace_id, RebuildRequest(), IndexResult, change)

    def search(self, query: str, *, limit: int = 20) -> tuple[dict[str, str | float], ...]:
        if not query.strip() or len(query) > 4096 or not 1 <= limit <= 100:
            raise ValueError("INVALID_SEARCH_BOUNDS")
        with self.authority.database.reader() as conn:
            rows = conn.execute(
                "SELECT c.source_id,c.title,c.snippet,bm25(source_fts) AS score FROM source_fts JOIN source_index c ON c.rowid=source_fts.rowid WHERE source_fts MATCH ? ORDER BY score LIMIT ?",
                (query, limit),
            ).fetchall()
            return tuple(
                {
                    "source_id": row["source_id"],
                    "title": row["title"],
                    "snippet": row["snippet"],
                    "score": row["score"],
                }
                for row in rows
            )
