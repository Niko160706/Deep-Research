"""Context Packer — assembles multi-store evidence (Python, 0 tokens).

Lifts the raw text OUT of parquet and computes (a) top keywords and (b) the
most query-relevant sentences, so the synthesis agent can write a narrative
instead of dumping URLs. Phase 2 adds the strict ``max_context_tokens`` cap.
"""
from __future__ import annotations

import logging
from urllib.parse import urlparse

from drx.store.graph_store import GraphStore
from drx.store.parquet_store import ParquetStore
from drx.store.vector_store import VectorStore
from drx.textutil import rank_sentences, top_keywords

log = logging.getLogger("drx.synthesis.packer")


def _domain(url: str) -> str:
    netloc = (urlparse(url).netloc or "").lower()
    return netloc.removeprefix("www.")


class ContextPacker:
    """Aggregates graph + vector + parquet stores into a narrative-ready context."""

    def build(
        self,
        query: str,
        graph_store: GraphStore,
        vector_store: VectorStore,
        parquet_store: ParquetStore,
    ) -> dict:
        documents = parquet_store.load_documents(limit=100)

        doc_meta = [
            {
                "url": d.url,
                "title": d.title,
                "domain": _domain(d.url),
            }
            for d in documents
        ]

        keywords = top_keywords(documents, n=15)
        sentences = rank_sentences(documents, query, n=12)

        return {
            "query": query,
            "documents": doc_meta,
            "keywords": keywords,        # list[(term, count)]
            "sentences": sentences,      # list[{text, source, title, score}]
            "graph": {
                "nodes": graph_store.node_count(),
                "edges": graph_store.edge_count(),
                "triples": graph_store.triples(limit=50),
            },
            "vector": {
                "cards": vector_store.card_count(),
                "top": vector_store.top_cards(k=10),
            },
        }
