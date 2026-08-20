"""
Entity handling for keyword selection.

Provides a lightweight, dependency-free way to parse a user-supplied entity
(person, brand, organization, location, concept), enrich search queries with
that entity, and prioritise keywords related to it.

The detection is heuristic (no external API required) so it works even when
spaCy / Ollama are unavailable. An explicit ``entity_type`` hint can override
the heuristic.
"""

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

ENTITY_TYPES = ["person", "brand", "organization", "location", "concept"]


class EntityHandler:
    """Parse, validate and apply an entity to keyword selection / search."""

    # Strong signals for each entity type used when no explicit type is given.
    _TYPE_HINTS = {
        "organization": [
            "inc",
            "corp",
            "corporation",
            "ltd",
            "llc",
            "company",
            "co.",
            "university",
            "agency",
            "foundation",
            "institute",
            "ministry",
        ],
        "brand": ["tm", "™", "®", "brand", "product", "model"],
        "location": [
            "city",
            "town",
            "country",
            "state",
            "island",
            "mountain",
            "river",
            "ocean",
            "sea",
            "street",
            "avenue",
            "park",
            "region",
            "province",
        ],
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def parse_entity(
        self, entity_string: Optional[str], entity_type: Optional[str] = None
    ) -> Optional[Dict]:
        """Parse + validate a raw entity string into a structured dict.

        Returns ``None`` when the input is empty/whitespace.
        """
        if not entity_string or not entity_string.strip():
            return None

        raw = entity_string.strip()
        detected = (
            entity_type if entity_type in ENTITY_TYPES else self._detect_type(raw)
        )
        tokens = [t for t in re.split(r"\s+", raw.lower()) if t]

        return {
            "raw": raw,
            "type": detected,
            "tokens": tokens,
            "keywords": self._extract_keywords(raw, tokens),
            "weight": 1.0,
        }

    def _detect_type(self, raw: str) -> str:
        low = raw.lower()
        for etype, hints in self._TYPE_HINTS.items():
            if any(h in low for h in hints):
                return etype
        # Heuristic: a single proper-noun-ish token -> concept; multiple
        # capitalised words often indicate a brand / organisation / person.
        words = raw.split()
        cap_words = [w for w in words if w[:1].isupper()]
        if len(words) >= 3:
            return "organization"
        if len(words) == 2 and len(cap_words) >= 1:
            return "brand"
        if len(words) == 1 and len(cap_words) == 1:
            return "concept"
        return "concept"

    def _extract_keywords(self, raw: str, tokens: List[str]) -> List[str]:
        """Derive a few keyword variants from the entity string."""
        keywords = [raw.lower()]
        if len(tokens) > 1:
            keywords.append(" ".join(tokens[:2]))
            keywords.append(tokens[-1])
        # De-duplicate while preserving order.
        seen = set()
        out = []
        for k in keywords:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------
    def enrich_query(self, base_query: str, entity: Optional[Dict]) -> str:
        """Enrich a search query with entity context (idempotent)."""
        if not entity:
            return base_query
        raw = entity.get("raw", "").strip()
        if not raw:
            return base_query
        # Avoid duplicating the entity if it is already in the query.
        if raw.lower() in base_query.lower():
            return base_query
        return f"{base_query} {raw}".strip()

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------
    def rank_keywords_by_entity(
        self, keywords: List[str], entity: Optional[Dict]
    ) -> List[str]:
        """Prioritise keywords that reference the entity.

        Keywords containing any entity token are boosted to the front while
        preserving relative order of the rest.
        """
        if not entity or not keywords:
            return keywords

        tokens = set(t for t in entity.get("tokens", []) if len(t) > 2)
        raw = entity.get("raw", "").lower()

        def score(kw: str) -> int:
            k = kw.lower()
            if k == raw:
                return 3
            if raw in k:
                return 2
            if tokens and any(t in k for t in tokens):
                return 1
            return 0

        return sorted(keywords, key=lambda k: (-score(k), keywords.index(k)))
