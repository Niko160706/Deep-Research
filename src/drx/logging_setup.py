"""Rich-based logging setup with a stdlib fallback.

`rich` is a base dependency, but we degrade gracefully so the pipeline can
still boot (and emit plain logs) in a minimal environment.
"""
from __future__ import annotations

import logging
import sys

try:  # pragma: no cover - branch depends on environment
    from rich.logging import RichHandler

    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _HAS_RICH = False


def setup_logging(verbose: bool = False) -> None:
    """Configure the ``drx`` logger once; idempotent across calls."""
    root = logging.getLogger("drx")
    if root.handlers:
        return

    level = logging.DEBUG if verbose else logging.INFO
    if _HAS_RICH:
        handler = RichHandler(rich_tracebacks=True, show_time=False, markup=False)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )

    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False

    # aiohttp is noisy at INFO; keep it quiet unless debugging
    if not verbose:
        logging.getLogger("aiohttp").setLevel(logging.WARNING)
