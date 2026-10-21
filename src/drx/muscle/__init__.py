"""The Muscle: 100% Python, 0 LLM tokens.

Sub-packages:
    backends    — pluggable search backends (DuckDuckGo HTML default)
    autosearch  — heuristic URL scoring + redundancy filtering
    scraper     — async aiohttp + Playwright fallback (Semaphore(2))
    cleaner     — HTML de-boilerplate + BM25 relevance trim
    chunker     — sliding-window text segmentation
"""
