"""Internship Hunter Engine — additive module for ``black-project``.

Public entry points::

    from drx.internship import InternshipEngine, run_internship

The package is intentionally independent from Mode 1.  It performs direct
job-board fetches (JobStreet, Glints, Kalibrr, ITS, MSIB, LinkedIn,
NodeFlair), logs source diagnostics, curates results and reuses the existing
Dark Neon Vis.js renderer for its mindmap.
"""
from __future__ import annotations

from .diagnostic_logger import (
    BLOCKED,
    EMPTY_RESULT,
    SUCCESS,
    TIMEOUT,
    DiagnosticLogger,
    SourceDiagnostic,
)
from .engine import InternshipEngine, InternshipRunResult, make_run_id, run_internship
from .models import CurationResult, InternshipJob, InternshipPlan, RawJob, SourceContent
from .site_parsers import AiohttpFetcher, MockFetcher, PlaywrightFetcher
from .wizard import InternshipWizardResult, prompt_internship_setup

__all__ = [
    "BLOCKED",
    "EMPTY_RESULT",
    "SUCCESS",
    "TIMEOUT",
    "AiohttpFetcher",
    "CurationResult",
    "DiagnosticLogger",
    "InternshipEngine",
    "InternshipJob",
    "InternshipPlan",
    "InternshipRunResult",
    "InternshipWizardResult",
    "MockFetcher",
    "PlaywrightFetcher",
    "RawJob",
    "SourceContent",
    "SourceDiagnostic",
    "make_run_id",
    "prompt_internship_setup",
    "run_internship",
]
