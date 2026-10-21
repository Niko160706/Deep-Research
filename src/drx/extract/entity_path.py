"""Entity track — fact extraction (keyword-based in Phase 1).

Phase 1 extracts meaningful keyword facts so the knowledge graph holds real
content words (not just ``title -> url``). Phase 2 wires ``deepseek-chat`` JSON
extraction (roles.extractor) and normalizes aliases with ``rapidfuzz``.
"""
from __future__ import annotations

import logging
import time

from drx.models import Document, Fact, SourceRef
from drx.textutil import top_keywords

log = logging.getLogger("drx.extract.entity")


class EntityExtractor:
    """Turns cleaned documents into keyword-anchored, provenance-bearing facts."""

    def __init__(self, keywords_per_doc: int = 3) -> None:
        self.keywords_per_doc = keywords_per_doc

    async def extract(self, docs: list[Document]) -> list[Fact]:
        facts: list[Fact] = []
        for doc in docs:
            ref = SourceRef(url=doc.url, title=doc.title or None, extracted_at=time.time())
            keywords = top_keywords([doc], n=self.keywords_per_doc)
            if not keywords:
                continue
            subject = doc.title or doc.url
            for keyword, _count in keywords:
                facts.append(
                    Fact(
                        subject=keyword,
                        predicate="disebut_dalam",
                        object=subject,
                        confidence_score=0.5,
                        provenance=[ref],
                    )
                )
        return facts
