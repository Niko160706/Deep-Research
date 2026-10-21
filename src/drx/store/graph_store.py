"""Knowledge graph storage backed by NetworkX (Phase 3, persistent).

Production upgrades over the Phase 1 store:

* **Stable node IDs** — every entity gets ``n1``, ``n2``, ... exactly once;
  the display label stays human-readable (``[n12] PT Jaya Abadi``) and the ID
  never changes across cycles, so the terminal checkpoint can pivot by node ID
  and the exported GraphML stays comparable between runs.
* **rapidfuzz entity resolution** — punctuation/prefix variants collapse onto
  one node (``PT Jaya Abadi`` == ``PT. Jaya Abadi``), with a strict ratio
  threshold plus a substring-length guard so a shorter name never swallows a
  longer different person.
* **Provenance on nodes AND edges** — source URLs, short quotes, confidence,
  ``source_count`` and ``verified`` (≥ 2 independent sources, per PRD_design
  edge styling). Edges also carry ``contradiction`` (functional-predicate
  conflicts), rendered red by the canvas.
* **Incremental export** — :meth:`export` can be called after every cycle
  (``runs/<run_id>/graph.graphml``).
"""
from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from pathlib import Path

from drx.models import Fact
from drx.textutil import name_tokens, tokenize

log = logging.getLogger("drx.store.graph")

try:  # pragma: no cover - dependency guard
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None  # type: ignore[assignment]

try:  # pragma: no cover - dependency guard
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover
    fuzz = None  # type: ignore[assignment]

#: Fuzzy ratio needed to merge two labels onto the same node.
RESOLUTION_THRESHOLD = 92.0
#: Prefix/degree/title tokens ignored when building the matching key.
_TITLE_TOKENS = {
    "ir", "dr", "drs", "dra", "h", "hj", "prof", "doktor", "s", "m",
    "sos", "kep", "si", "pd", "kom", "farm", "apt", "tk", "mm", "mh",
}
_ORG_COMPANY_PREFIX = ("pt", "cv", "ud", "pd", "firma", "fa", "perum", "persero")
_ORG_MARKERS = (
    "dinas", "kementerian", "kementrian", "badan", "lembaga", "kantor",
    "universitas", "institut", "sekolah", "pemerintah", "pemprov", "pemkab",
    "pemkot", "kejaksaan", "pengadilan", "polres", "polda", "kodam", "bumn",
    "bank", "yayasan", "organisasi", "asosiasi", "komisi", "dewan", "forum",
    "puskesmas", "rumah sakit", "rsud", "kecamatan", "kelurahan", "desa",
)
_DOCUMENT_MARKERS = (
    "sk ", "surat keputusan", "lhkpn", "berita acara", "dokumen", "sertifikat",
    "ijazah", "akta", "kontrak", "laporan", "nota dinas", "undangan",
)
_EVENT_MARKERS = (
    "pelantikan", "rapat", "acara", "kegiatan", "pernikahan", "pemilu",
    "seleksi", "rekrutmen", "pengadaan", "kunjungan", "sosialisasi",
)

#: Predicates that may only have ONE object per subject; a second, different
#: object is a contradiction (rendered red by the canvas + flagged by critic).
_FUNCTIONAL_PREDICATES = {
    "bekerja_sebagai", "bekerja_di", "jabatan", "menjabat", "lahir_di",
    "tanggal_lahir", "menikah_dengan", "status_perkawinan", "nip", "nik",
    "alamat", "istri", "suami", "anak", "ibu", "ayah",
}

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_name(name: str) -> str:
    """Normalization key: NFKC, casefold, punctuation-free, title-free."""
    flat = unicodedata.normalize("NFKC", name or "").casefold()
    flat = _PUNCT_RE.sub(" ", flat)
    tokens = [t for t in _WS_RE.sub(" ", flat).strip().split(" ") if t]
    kept = [t for t in tokens if t not in _TITLE_TOKENS]
    return " ".join(kept or tokens)


def infer_node_type(name: str) -> str:
    """Coarse entity type used for canvas badges/colors (PRD_design §2.1)."""
    low = f" {(name or '').casefold()} "
    tokens = normalize_name(name).split()
    if any(marker in low for marker in _DOCUMENT_MARKERS):
        return "dokumen"
    if any(marker in low for marker in _EVENT_MARKERS):
        return "peristiwa"
    if tokens and tokens[0] in _ORG_COMPANY_PREFIX:
        return "perusahaan"
    if any(marker in low for marker in _ORG_MARKERS):
        return "instansi"
    words = [w for w in (name or "").split() if w[:1].isupper() or w[:1].isdigit()]
    if 1 <= len(words) <= 4 and not any(ch.isdigit() for ch in name or ""):
        return "individu"
    return "lainnya"


def _norm_predicate(predicate: str) -> str:
    flat = normalize_name(predicate)
    return flat.replace(" ", "_")


class GraphStore:
    """In-memory multi-digraph with stable IDs + GraphML export."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.graph = nx.MultiDiGraph() if nx is not None else None
        self._known_nodes: set[str] = set()
        self._by_key: dict[str, str] = {}
        self._next_id = 1

    # ---- node/edge helpers --------------------------------------------------

    def _new_node_id(self) -> str:
        node_id = f"n{self._next_id}"
        self._next_id += 1
        return node_id

    def label_of(self, node: object) -> str:
        """Display label for a node id (falls back to the raw key)."""
        if self.graph is not None and node in self.graph:
            return str(self.graph.nodes[node].get("label") or node)
        return str(node)

    def type_of(self, node: object) -> str:
        if self.graph is not None and node in self.graph:
            return str(self.graph.nodes[node].get("type") or "lainnya")
        if isinstance(node, str) and node.startswith("n") and node[1:].isdigit():
            return "lainnya"
        return infer_node_type(str(node))

    def _register_key(self, key: str, node_id: str) -> None:
        if key and key not in self._by_key:
            self._by_key[key] = node_id

    def _match_existing(self, name: str) -> str | None:
        """rapidfuzz resolution against known normalized keys (type-aware)."""
        key = normalize_name(name)
        if not key:
            return None
        existing = self._by_key.get(key)
        if existing is not None and existing in self._known_nodes:
            return existing
        if fuzz is None:
            return None
        node_type = infer_node_type(name)
        best_id: str | None = None
        best_ratio = 0.0
        for other_key, node_id in self._by_key.items():
            if node_id not in self._known_nodes:
                continue
            other_type = self.type_of(node_id)
            same_org_family = {node_type, other_type} <= {"instansi", "perusahaan"}
            if (
                node_type != "lainnya"
                and other_type != "lainnya"
                and node_type != other_type
                and not same_org_family
            ):
                continue
            ratio = float(fuzz.ratio(key, other_key))
            if key in other_key or other_key in key:
                # Prevent "sri lestari" swallowing "sri lestari wulandari".
                if abs(len(key) - len(other_key)) > 3:
                    ratio = min(ratio, 80.0)
            if ratio > best_ratio:
                best_ratio = ratio
                best_id = node_id
        if best_id is not None and best_ratio >= RESOLUTION_THRESHOLD:
            self._register_key(key, best_id)
            aliases = self.graph.nodes[best_id].setdefault("aliases", [])
            if name not in aliases and normalize_name(name) != normalize_name(
                self.label_of(best_id)
            ):
                aliases.append(name)
            return best_id
        return None

    def _resolve(self, name: str, level: int = 1) -> tuple[str, bool]:
        """Return ``(node_id, created)`` for a display name.

        ``level`` is the exploration hop recorded on *new* nodes only:
        ``0`` root query, ``1`` initial search, ``2`` pivot round 1,
        ``3+`` subsequent pivot rounds (used by the hierarchical canvas).
        Existing nodes keep the level they were first discovered at.
        """
        assert self.graph is not None
        clean = _WS_RE.sub(" ", (name or "").strip()) or "(tanpa nama)"
        matched = self._match_existing(clean)
        if matched is not None:
            return matched, False
        node_id = self._new_node_id()
        key = normalize_name(clean)
        self._register_key(key, node_id)
        self.graph.add_node(
            node_id,
            label=clean,
            normalized=key,
            type=infer_node_type(clean),
            aliases=[],
            sources=[],
            quotes=[],
            confidence=0.0,
            source_count=0,
            verified=False,
            level=max(0, int(level)),
            created_at=time.time(),
        )
        self._known_nodes.add(node_id)
        return node_id, True

    def ensure_root(self, name: str) -> str | None:
        """Create/find the root query node and pin it to ``level=0``.

        Called once during ``_setup`` so the hierarchical mindmap always has a
        single left-most anchor even before the first fact mentions the target.
        """
        if self.graph is None:
            return None
        clean = _WS_RE.sub(" ", (name or "").strip())
        if not clean:
            return None
        node_id, _created = self._resolve(clean, level=0)
        self.graph.nodes[node_id]["level"] = 0
        self.graph.nodes[node_id]["root"] = True
        return node_id

    @staticmethod
    def _merge_provenance(attrs: dict, fact: Fact) -> None:
        sources = attrs.setdefault("sources", [])
        quotes = attrs.setdefault("quotes", [])
        for ref in fact.provenance:
            if ref.url and ref.url not in sources:
                sources.append(ref.url)
            quote = (getattr(ref, "quote", "") or "").strip()
            if quote and quote not in quotes:
                quotes.append(quote[:400])
        attrs["source_count"] = len(sources)
        attrs["verified"] = attrs["source_count"] >= 2
        attrs["confidence"] = max(
            float(attrs.get("confidence") or 0.0), float(fact.confidence_score)
        )

    def _upsert_edge(self, subject_id: str, object_id: str, fact: Fact) -> None:
        assert self.graph is not None
        predicate = fact.predicate or "terkait"
        norm_pred = _norm_predicate(predicate)
        for u, v, key, data in list(
            self.graph.edges(subject_id, keys=True, data=True)
        ):
            if v == object_id and data.get("norm_predicate") == norm_pred:
                merged = dict(data)
                self._merge_provenance(merged, fact)
                self.graph.remove_edge(u, v, key=key)
                self.graph.add_edge(u, v, key=key, **merged)
                return

        attrs: dict = {
            "predicate": predicate,
            "norm_predicate": norm_pred,
            "target": object_id,
            "sources": [],
            "quotes": [],
            "source_count": 0,
            "verified": False,
            "contradiction": False,
        }
        self._merge_provenance(attrs, fact)
        self.graph.add_edge(subject_id, object_id, **attrs)

        if norm_pred in _FUNCTIONAL_PREDICATES:
            conflicting: list[tuple[str, str, int]] = []
            for u, v, key, data in list(
                self.graph.edges(keys=True, data=True)
            ):
                if data.get("norm_predicate") != norm_pred:
                    continue
                if (u == subject_id and v != object_id) or (
                    u == object_id and v != subject_id
                ):
                    conflicting.append((u, v, key))
            if conflicting:
                # Flag both the old and the new claim so the canvas renders
                # every conflicting edge red (PRD §2.1.2).
                for u, v, key in conflicting:
                    self.graph.edges[u, v, key]["contradiction"] = True
                for u, v, key, data in list(
                    self.graph.edges(subject_id, keys=True, data=True)
                ):
                    if (
                        v == object_id
                        and data.get("norm_predicate") == norm_pred
                    ):
                        data["contradiction"] = True

    # ---- public write API ----------------------------------------------------

    def resolve_and_add(self, facts: list[Fact], *, level: int = 1) -> int:
        """Add facts, returning the count of NEWLY created nodes.

        ``level`` labels the exploration hop for *new* nodes (see
        :meth:`_resolve`); nodes already present keep their earlier level.
        """
        if self.graph is None:
            log.warning("networkx unavailable; graph store inert")
            return 0
        new_nodes = 0
        for fact in facts:
            subject_id, created_s = self._resolve(fact.subject, level=level)
            object_id, created_o = self._resolve(fact.object, level=level)
            new_nodes += int(created_s) + int(created_o)
            for node_id in (subject_id, object_id):
                self._merge_provenance(self.graph.nodes[node_id], fact)
            self._upsert_edge(subject_id, object_id, fact)
        return new_nodes

    # ---- introspection -------------------------------------------------------

    def node_count(self) -> int:
        return self.graph.number_of_nodes() if self.graph else 0

    def edge_count(self) -> int:
        return self.graph.number_of_edges() if self.graph else 0

    def triples(self, limit: int = 50) -> list[dict]:
        """Compact triple JSON used by the critic (NO raw text leaves disk).

        ``limit <= 0`` returns every edge (used by evidence-only synthesis so a
        deep interactive graph is not silently truncated at 500 facts).
        """
        if not self.graph:
            return []
        edges = list(self.graph.edges(data=True))
        if limit and limit > 0:
            edges = edges[:limit]
        out: list[dict] = []
        for u, v, data in edges:
            out.append(
                {
                    "subject": self.label_of(u),
                    "predicate": data.get("predicate"),
                    "object": self.label_of(v),
                    "confidence": data.get("confidence"),
                    "subject_id": u,
                    "object_id": v,
                    "sources": list(data.get("sources") or []),
                    "quotes": list(data.get("quotes") or [])[:2],
                    "source_count": int(data.get("source_count") or 0),
                    "verified": bool(data.get("verified")),
                    "contradiction": bool(data.get("contradiction")),
                }
            )
        return out

    def node_records(self, limit: int = 0) -> list[dict]:
        """Canvas payload: one record per node (provenance included)."""
        if not self.graph:
            return []
        records: list[dict] = []
        nodes = list(self.graph.nodes(data=True))
        if limit and limit > 0:
            nodes = nodes[:limit]
        for node_id, attrs in nodes:
            records.append(
                {
                    "id": node_id,
                    "label": str(attrs.get("label") or node_id),
                    "type": str(attrs.get("type") or "lainnya"),
                    "aliases": list(attrs.get("aliases") or []),
                    "sources": list(attrs.get("sources") or []),
                    "quotes": list(attrs.get("quotes") or [])[:3],
                    "source_count": int(attrs.get("source_count") or 0),
                    "verified": bool(attrs.get("verified")),
                    "confidence": float(attrs.get("confidence") or 0.0),
                    "degree": int(self.graph.degree(node_id)),
                    "level": max(0, int(attrs.get("level", 1))),
                    "root": bool(attrs.get("root")),
                }
            )
        return records

    def edge_records(self, limit: int = 0) -> list[dict]:
        if not self.graph:
            return []
        records: list[dict] = []
        edges = list(self.graph.edges(data=True))
        if limit and limit > 0:
            edges = edges[:limit]
        for u, v, data in edges:
            records.append(
                {
                    "source": u,
                    "target": v,
                    "predicate": str(data.get("predicate") or ""),
                    "sources": list(data.get("sources") or []),
                    "quotes": list(data.get("quotes") or [])[:2],
                    "source_count": int(data.get("source_count") or 0),
                    "verified": bool(data.get("verified")),
                    "contradiction": bool(data.get("contradiction")),
                    "confidence": float(data.get("confidence") or 0.0),
                }
            )
        return records

    def neighbors(self, node_id: str, limit: int = 6) -> list[dict]:
        """Neighbor context for a pivot query (anti name-collision guard)."""
        if not self.graph or node_id not in self.graph:
            return []
        out: list[dict] = []
        for _, target, data in self.graph.out_edges(node_id, data=True):
            out.append(
                {
                    "node_id": target,
                    "label": self.label_of(target),
                    "predicate": data.get("predicate") or "",
                    "direction": "out",
                }
            )
        for source, _, data in self.graph.in_edges(node_id, data=True):
            out.append(
                {
                    "node_id": source,
                    "label": self.label_of(source),
                    "predicate": data.get("predicate") or "",
                    "direction": "in",
                }
            )
        seen: set[str] = set()
        unique: list[dict] = []
        for item in out:
            if item["node_id"] in seen:
                continue
            seen.add(item["node_id"])
            unique.append(item)
        return unique[:limit]

    def suggest_nodes(
        self,
        target_entity: str = "",
        context: str = "",
        *,
        limit: int = 5,
        exclude: set[str] | None = None,
    ) -> list[dict]:
        """Deterministic fallback suggestions when the LLM critic is unavailable.

        Score = degree + lexical overlap with target/context; the target's own
        node is excluded, and generic/heuristic nodes rank lower.
        """
        if not self.graph:
            return []
        exclude = set(exclude or ())
        target_key = normalize_name(target_entity)
        context_tokens = set(tokenize(context)) | {t.lower() for t in name_tokens(target_entity)}
        scored: list[tuple[float, str]] = []
        for node_id, attrs in self.graph.nodes(data=True):
            if node_id in exclude:
                continue
            label = str(attrs.get("label") or node_id)
            key = normalize_name(label)
            if target_key and (key == target_key or target_key in key):
                continue
            degree = float(self.graph.degree(node_id))
            overlap = 0.0
            tokens = {t.lower() for t in tokenize(label)}
            if context_tokens:
                overlap = len(tokens & context_tokens) / len(context_tokens)
            type_bonus = {"individu": 0.6, "instansi": 0.5, "perusahaan": 0.5,
                          "dokumen": 0.3, "peristiwa": 0.3}.get(
                str(attrs.get("type") or "lainnya"), 0.0
            )
            verified_bonus = 0.5 if attrs.get("verified") else 0.0
            scored.append((degree + 2.0 * overlap + type_bonus + verified_bonus, node_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        suggestions: list[dict] = []
        for score, node_id in scored[: max(0, limit)]:
            suggestions.append(
                {
                    "node_id": node_id,
                    "label": self.label_of(node_id),
                    "reason": (
                        f"derajat koneksi {self.graph.degree(node_id)}, "
                        f"skor {score:.1f}"
                    ),
                }
            )
        return suggestions

    # ---- pruning / export ------------------------------------------------------

    def _target_match_count(self, community: set, target: str) -> int:
        target_low = target.lower()
        target_tokens = {t.lower() for t in tokenize(target)}
        target_tokens |= {t.lower() for t in name_tokens(target)}
        target_tokens.add(target_low)
        count = 0
        for node in community:
            label = self.label_of(node).lower()
            if (
                str(node) == target
                or label == target_low
                or label in target_tokens
                or any(t in label for t in target_tokens if len(t) > 3)
            ):
                count += 1
        return count

    def prune_louvain(self, target: str | None = None) -> int:
        """Prune isolated communities via Louvain, keeping the main cluster
        (containing ``target`` when given) plus any cluster bridged to it by an
        edge. Returns the number of removed nodes."""
        if self.graph is None or self.graph.number_of_nodes() < 2:
            return 0
        try:
            from networkx.algorithms.community import louvain_communities
        except ImportError:  # pragma: no cover
            log.warning("louvain unavailable; skipping pruning")
            return 0
        try:
            communities = list(louvain_communities(self.graph, seed=42))
        except Exception as exc:
            log.warning("louvain failed: %s", exc)
            return 0
        if not communities:
            return 0

        # The level-0 root anchor (GraphStore.ensure_root) is a UI pin, not a
        # fact community. If it is still isolated it must never win "main
        # cluster" selection and cause the real fact cluster to be pruned.
        root_nodes = {
            node
            for node, attrs in self.graph.nodes(data=True)
            if attrs.get("root")
        }
        isolated_roots = {
            node for node in root_nodes if self.graph.degree(node) == 0
        }
        selectable = [
            community
            for community in communities
            if not (isolated_roots and set(community) <= isolated_roots)
        ] or communities

        main = None
        if target:
            target_low = target.strip().lower()
            best = max(
                selectable,
                key=lambda c: (
                    # komunitas yang memuat node target persis selalu menang,
                    # baru kemudian jumlah kecocokan dan ukuran komunitas
                    sum(
                        1
                        for node in c
                        if self.label_of(node).strip().lower() == target_low
                    ),
                    self._target_match_count(c, target),
                    len(c),
                ),
            )
            if self._target_match_count(best, target) > 0:
                main = best
        if main is None:
            main = max(selectable, key=len)

        main_set = set(main)
        keep = set(main_set) | root_nodes
        for comm in communities:
            comm_set = set(comm)
            if comm_set == main_set:
                continue
            # keep a cluster if it has a bridge edge into the main cluster
            bridged = any(
                self.graph.has_edge(u, v) or self.graph.has_edge(v, u)
                for u in comm_set
                for v in main_set
            )
            if bridged:
                keep |= comm_set

        removed = 0
        for node in list(self.graph.nodes()):
            if node not in keep:
                self.graph.remove_node(node)
                self._known_nodes.discard(node)
                removed += 1
        if removed:
            self._reindex()
            log.info("louvain pruning removed %d isolated nodes", removed)
        return removed

    def _reindex(self) -> None:
        """Rebuild normalized-key → id map after removals/relabels."""
        assert self.graph is not None
        self._known_nodes = set(self.graph.nodes())
        self._by_key = {}
        for node_id, attrs in self.graph.nodes(data=True):
            self._register_key(str(attrs.get("normalized") or normalize_name(self.label_of(node_id))), node_id)
            for alias in attrs.get("aliases") or []:
                self._register_key(normalize_name(str(alias)), node_id)

    def export(self) -> Path | None:
        """Incremental GraphML export (safe to call after every cycle).

        GraphML only supports scalar attributes, so list/dict provenance is
        JSON-encoded on a shallow copy — the in-memory graph keeps its rich
        Python values for the canvas and the critic.
        """
        if not self.graph:
            return None
        path = self.run_dir / "graph.graphml"
        try:
            export_graph = self._exportable_copy()
            nx.write_graphml(export_graph, str(path))
            return path
        except Exception as exc:
            log.warning("graphml export failed: %s", exc)
            return None

    def _exportable_copy(self):
        assert self.graph is not None
        clone = self.graph.__class__()
        for node_id, attrs in self.graph.nodes(data=True):
            clone.add_node(node_id, **self._scalarize(attrs))
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            clone.add_edge(u, v, key=key, **self._scalarize(data))
        return clone

    @staticmethod
    def _scalarize(attrs: dict) -> dict:
        out: dict = {}
        for key, value in attrs.items():
            if isinstance(value, (list, dict, tuple, set)):
                out[key] = json.dumps(list(value), ensure_ascii=False, default=str)
            elif value is None:
                out[key] = ""
            else:
                out[key] = value
        return out
