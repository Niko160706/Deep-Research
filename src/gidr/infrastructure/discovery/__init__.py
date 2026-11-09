"""Metadata-first, capability-aware source discovery."""

from gidr.infrastructure.discovery.adapters import P0_ADAPTERS, DiscoveryBatch, DiscoveryContext
from gidr.infrastructure.discovery.contracts import (
    CanonicalQuery,
    FilterDisposition,
    ProviderCapabilities,
    ProviderHealth,
    SourceCoverageObservation,
)
from gidr.infrastructure.discovery.registry import ProviderRegistry, default_registry

__all__ = [
    "P0_ADAPTERS",
    "CanonicalQuery",
    "DiscoveryBatch",
    "DiscoveryContext",
    "FilterDisposition",
    "ProviderCapabilities",
    "ProviderHealth",
    "ProviderRegistry",
    "SourceCoverageObservation",
    "default_registry",
]
