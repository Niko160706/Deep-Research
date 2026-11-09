"""Explicit provider catalogue and safe status reporting."""

from __future__ import annotations

from collections.abc import Iterable

from gidr.infrastructure.discovery.contracts import ProviderCapabilities


class ProviderRegistry:
    def __init__(self, providers: Iterable[ProviderCapabilities] = ()) -> None:
        self._providers: dict[str, ProviderCapabilities] = {}
        for provider in providers:
            self.register(provider)

    def register(self, provider: ProviderCapabilities) -> None:
        if provider.provider_id in self._providers:
            raise ValueError("DUPLICATE_PROVIDER_ID")
        self._providers[provider.provider_id] = provider

    def get(self, provider_id: str) -> ProviderCapabilities:
        try:
            return self._providers[provider_id]
        except KeyError:
            raise KeyError("PROVIDER_NOT_REGISTERED") from None

    def all(self) -> tuple[ProviderCapabilities, ...]:
        return tuple(self._providers[key] for key in sorted(self._providers))

    def implemented(self) -> tuple[ProviderCapabilities, ...]:
        return tuple(p for p in self.all() if p.status != "NOT_IMPLEMENTED")

    def ordered_for(self, categories: tuple[str, ...]) -> tuple[ProviderCapabilities, ...]:
        selected = (
            p
            for p in self._providers.values()
            if p.category in categories and p.status != "NOT_IMPLEMENTED"
        )
        return tuple(sorted(selected, key=lambda p: (-p.priority, p.provider_id)))


_P0: tuple[ProviderCapabilities, ...] = (
    ProviderCapabilities(
        provider_id="searxng",
        category="web",
        priority=100,
        status="MOCK_TESTED",
        native_filters=("text",),
        local_filters=("language", "domain", "date", "required_terms"),
        pagination="PAGE",
        max_page_size=20,
        base_hosts=(),
        status_reason="CONFIGURED_INSTANCE_REQUIRED",
    ),
    ProviderCapabilities(
        provider_id="gdelt",
        category="web",
        priority=85,
        status="MOCK_TESTED",
        native_filters=("text", "date"),
        local_filters=("domain", "language", "required_terms"),
        pagination="NONE",
        max_page_size=100,
        base_hosts=("api.gdeltproject.org",),
    ),
    ProviderCapabilities(
        provider_id="wayback_cdx",
        category="archive",
        priority=40,
        status="MOCK_TESTED",
        native_filters=("text", "date"),
        local_filters=("domain", "required_terms"),
        pagination="NONE",
        max_page_size=100,
        base_hosts=("web.archive.org",),
    ),
    ProviderCapabilities(
        provider_id="arxiv",
        category="academic",
        priority=70,
        status="MOCK_TESTED",
        native_filters=("text",),
        local_filters=("date", "domain", "required_terms"),
        pagination="OFFSET",
        max_page_size=100,
        base_hosts=("export.arxiv.org",),
    ),
    ProviderCapabilities(
        provider_id="openalex",
        category="academic",
        priority=80,
        status="MOCK_TESTED",
        native_filters=("text", "date"),
        local_filters=("language", "domain", "required_terms"),
        pagination="CURSOR",
        max_page_size=100,
        base_hosts=("api.openalex.org",),
    ),
    ProviderCapabilities(
        provider_id="crossref",
        category="academic",
        priority=75,
        status="MOCK_TESTED",
        native_filters=("text", "date"),
        local_filters=("language", "domain", "required_terms"),
        pagination="OFFSET",
        max_page_size=100,
        base_hosts=("api.crossref.org",),
    ),
    ProviderCapabilities(
        provider_id="zenodo",
        category="data",
        priority=60,
        status="MOCK_TESTED",
        native_filters=("text", "date"),
        local_filters=("language", "domain", "required_terms"),
        pagination="PAGE",
        max_page_size=100,
        base_hosts=("zenodo.org",),
    ),
    ProviderCapabilities(
        provider_id="github",
        category="code",
        priority=50,
        status="MOCK_TESTED",
        native_filters=("text", "date"),
        local_filters=("domain", "required_terms"),
        pagination="PAGE",
        max_page_size=30,
        metadata_first=True,
        default_clone=False,
        base_hosts=("api.github.com",),
    ),
)

_OPTIONAL_CATEGORIES = {
    "semantic_scholar": "academic",
    "core": "academic",
    "doaj": "academic",
    "osf": "data",
    "figshare": "data",
    "datacite": "data",
    "ahu_online": "institutional",
    "data_go_id": "institutional",
    "data_gov": "institutional",
    "kaggle": "code_data",
    "hugging_face": "code_data",
    "internet_archive_collections": "archive",
    "open_library": "archive",
    "europeana": "archive",
    "gutenberg": "archive",
}


def default_registry() -> ProviderRegistry:
    registry = ProviderRegistry(_P0)
    for provider_id, category in _OPTIONAL_CATEGORIES.items():
        registry.register(
            ProviderCapabilities(
                provider_id=provider_id,
                category=category,
                priority=0,
                status="NOT_IMPLEMENTED",
                native_filters=(),
                pagination="NONE",
                max_page_size=1,
                base_hosts=(),
                status_reason="CONDITIONAL_PROVIDER_NOT_IMPLEMENTED",
            )
        )
    return registry
