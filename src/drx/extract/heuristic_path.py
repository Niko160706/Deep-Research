"""Heuristic track — argument cards + INT8 embedding.

Phase 1 chunks text into argument cards (mechanical, functional) and treats
embedding as a no-op. Phase 2 wires the BGE-small INT8 encoder and pushes
vectors into the FAISS-backed :class:`VectorStore`.
"""
from __future__ import annotations

import logging

from drx.models import Document, EvidenceCard
from drx.muscle.chunker import Chunker

log = logging.getLogger("drx.extract.heuristic")


class HeuristicExtractor:
    """Segments documents into structured argument cards."""

    def __init__(self, chunker: Chunker) -> None:
        self.chunker = chunker

    def make_cards(self, docs: list[Document]) -> list[EvidenceCard]:
        cards: list[EvidenceCard] = []
        for doc in docs:
            for i, chunk in enumerate(self.chunker.chunk_document(doc)):
                cards.append(
                    EvidenceCard(
                        chunk_id=f"{doc.url}#{i}",
                        url=doc.url,
                        claim=chunk[:200],
                        evidence=chunk,
                    )
                )
        return cards

    async def embed(self, cards: list[EvidenceCard]) -> list[EvidenceCard]:
        # Mock — Phase 2 wires BGE-small INT8 + FAISS.
        return cards
