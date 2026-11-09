"""Provider-specific filter accounting; mandatory unsupported constraints stop dispatch."""

from gidr.infrastructure.discovery.contracts import (
    CanonicalQuery,
    CompiledFilter,
    CompiledQuery,
    FilterDisposition,
    ProviderCapabilities,
)


def compile_query(query: CanonicalQuery, capabilities: ProviderCapabilities) -> CompiledQuery:
    requested = {
        "text": bool(query.text),
        "date": query.date_from is not None or query.date_to is not None,
        "language": bool(query.languages),
        "domain": bool(query.domains),
        "required_terms": bool(query.required_terms),
    }
    mandatory = set(query.mandatory_filters) | {"text"}
    filters: list[CompiledFilter] = []
    for name, present in requested.items():
        if not present:
            continue
        disposition: FilterDisposition
        if name in capabilities.native_filters:
            disposition = "NATIVE"
            reason = None
        elif name in capabilities.local_filters:
            disposition = "LOCAL_POSTFILTER"
            reason = "PROVIDER_FILTER_UNAVAILABLE"
        else:
            disposition = "UNSUPPORTED"
            reason = "PROVIDER_FILTER_UNSUPPORTED"
        filters.append(
            CompiledFilter(
                name=name,
                disposition=disposition,
                mandatory=name in mandatory,
                reason=reason,
            )
        )
    blocked = any(f.mandatory and f.disposition == "UNSUPPORTED" for f in filters)
    return CompiledQuery(
        provider_id=capabilities.provider_id,
        original=query,
        filters=tuple(filters),
        blocked_reason="MANDATORY_FILTER_UNSUPPORTED" if blocked else None,
    )
