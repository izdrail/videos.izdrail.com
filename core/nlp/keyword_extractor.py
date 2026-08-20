"""
Keyword Extraction
Uses AI and NLP to extract visual keywords from text
"""

import logging
import os
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import requests
import spacy

from .entity import EntityHandler
from .neuron_extractor import NeuronExtractor
from .ollama_client import (
    DEFAULT_FALLBACK_KEYWORDS,
    DEFAULT_FALLBACK_MOOD,
    DEFAULT_FALLBACK_SCRIPT,
    OllamaClient,
)

logger = logging.getLogger(__name__)


class OllamaKeywordExtractor:
    """Uses Ollama API to extract keywords from text"""

    def __init__(self, model: str = "gemma4:e2b", url: Optional[str] = None):
        self.model = model
        self.url = url or os.getenv(
            "OLLAMA_API_URL", "https://ai.izdrail.com/api/generate"
        )
        self.cache = {}

        # Build resilient client from environment / defaults
        self._client = OllamaClient(
            model=self.model,
            url=self.url,
            max_retries=int(os.getenv("OLLAMA_MAX_RETRIES", "3")),
            base_delay=float(os.getenv("OLLAMA_RETRY_BASE_DELAY", "1.0")),
            timeout=int(os.getenv("OLLAMA_TIMEOUT", "180")),
            cache_max_size=int(os.getenv("OLLAMA_CACHE_MAX_SIZE", "512")),
        )

    # ------------------------------------------------------------------
    # Sync client state when model/url are changed externally
    # ------------------------------------------------------------------
    def _sync_client(self) -> None:
        self._client.set_model(self.model)
        self._client.set_url(self.url)

    # ------------------------------------------------------------------
    # Keyword extraction
    # ------------------------------------------------------------------

    def extract_keywords(
        self,
        text: str,
        top_n: int = 5,
        language: str = "en",
        theme: Optional[str] = None,
    ) -> List[str]:
        cache_key = f"{text[:100]}_{top_n}_{language}_{theme}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        theme_line = ""
        if theme:
            theme_line = (
                f"- The overarching theme of the whole script is '{theme}'. "
                f"Favor keywords that reinforce this theme.\n"
            )

        prompt = (
            f"You are a stock footage search expert. Extract {top_n} visual keywords from the text below. "
            f"Rules:\n"
            f"- Each keyword must be 1-2 words maximum\n"
            f"- Prioritize: physical objects, locations, actions, nature, technology, people activities\n"
            f"- Use common stock footage terms (e.g., 'city skyline', 'ocean waves', 'forest', 'office')\n"
            f"- Avoid abstract concepts unless they have clear visual representations\n"
            f"{theme_line}"
            f"- Return ONLY a comma-separated list in {language}\n\n"
            f'Text: "{text}"\n'
            f"Keywords:"
        )
        self._sync_client()
        result = self._client.post_or_fallback(
            prompt,
            fallback=None,
            options={"temperature": 0.3, "num_predict": 64},
            timeout=180,
        )

        if result is None:
            logger.warning("[Ollama] All retries failed — using fallback keywords")
            fallback_kws = [
                kw.strip()
                for kw in os.getenv(
                    "OLLAMA_FALLBACK_KEYWORDS",
                    ",".join(DEFAULT_FALLBACK_KEYWORDS),
                ).split(",")
                if kw.strip()
            ]
            return fallback_kws[:top_n]

        raw = result.get("response", "").strip()
        logger.debug("[Ollama] Raw output: %s", raw)

        raw_keywords = [kw.strip().lower() for kw in raw.split(",") if kw.strip()]
        keywords = [re.sub(r"[^a-zA-Z0-9\s]", "", kw) for kw in raw_keywords]

        filtered_keywords = []
        for kw in keywords:
            words = kw.split()
            if len(words) == 1 and words[0]:
                filtered_keywords.append(words[0])
            elif len(words) >= 2:
                filtered_keywords.append(f"{words[0]} {words[1]}")

        out = filtered_keywords[:top_n]
        self.cache[cache_key] = out
        return out

    # ------------------------------------------------------------------
    # Theme extraction
    # ------------------------------------------------------------------

    def extract_theme(self, text: str, language: str = "en") -> Optional[str]:
        """Extract a single overarching theme (1-3 words) for the whole script."""
        cache_key = f"theme_{language}_{text[:200]}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = (
            f"You are a narrative analyst. Identify the single overarching theme of the "
            f"following text in 1-3 words (a noun phrase, not a full sentence). "
            f"This theme will guide stock-footage keyword selection. "
            f"Respond with ONLY the theme phrase in {language}.\n\n"
            f'Text: "{text[:1500]}"\n'
            f"Theme:"
        )
        self._sync_client()
        result = self._client.post_or_fallback(
            prompt,
            fallback=None,
            options={"temperature": 0.2, "num_predict": 24},
            timeout=60,
        )
        theme = None
        if result is not None:
            raw = result.get("response", "").strip()
            theme = re.sub(r"[^a-zA-Z0-9\s]", "", raw).strip().lower()
            theme = " ".join(theme.split()[:3]) if theme else None

        self.cache[cache_key] = theme
        return theme

    # ------------------------------------------------------------------
    # Social media descriptions
    # ------------------------------------------------------------------

    def generate_social_media_descriptions(
        self, text: str, keywords: List[str], language: str = "en"
    ) -> str:
        prompt = f"""
        You are a professional social media manager.
        Based on the following video script and extracted keywords, create:
        1. A catchy YouTube Video Title (max 60 chars)
        2. A compelling Video Description (max 200 chars)
        3. A list of 10 relevant hashtags
        4. A short TikTok/Reels caption (max 100 chars)

        Script: "{text[:1000]}..."
        Keywords: {", ".join(keywords)}
        Language: {language}

        Output format:
        Title: [Title]
        Description: [Description]
        Hashtags: #tag1 #tag2 ...
        TikTok: [Caption]
        """
        self._sync_client()
        result = self._client.post_or_fallback(
            prompt,
            fallback=None,
            options={"temperature": 0.7},
            timeout=30,
        )
        if result is None:
            return "Failed to generate descriptions (Ollama unavailable)."
        return result.get("response", "").strip() or "Failed to generate descriptions."

    # ------------------------------------------------------------------
    # Mood extraction
    # ------------------------------------------------------------------

    def extract_mood_keyword(self, text: str) -> str:
        prompt = (
            f"Analyze the emotional tone of the text below and return ONLY ONE word "
            f"representing a musical mood or genre that fits as background music. "
            f"Examples: Epic, Relaxing, Cinematic, Happy, Dark, Energetic, Lo-fi.\n\n"
            f'Text: "{text[:500]}"\n'
            f"Mood:"
        )
        self._sync_client()
        result = self._client.post_or_fallback(
            prompt,
            fallback=None,
            options={"temperature": 0.4, "num_predict": 10},
            timeout=30,
        )
        if result is None:
            logger.warning("[Ollama] Mood extraction failed — using fallback")
            return os.getenv("OLLAMA_FALLBACK_MOOD", DEFAULT_FALLBACK_MOOD)

        mood = result.get("response", "").strip().split()[0]
        return re.sub(r"[^a-zA-Z]", "", mood).capitalize() or os.getenv(
            "OLLAMA_FALLBACK_MOOD", DEFAULT_FALLBACK_MOOD
        )

    # ------------------------------------------------------------------
    # Script generation
    # ------------------------------------------------------------------

    def generate_script_from_text(self, text: str) -> str:
        prompt = (
            "Generate a TTS-ready script from the following text. "
            "Remove [pause] tags, stage directions, and any non-spoken instructions. "
            "Return only the spoken content:\n\n"
            f"{text}"
        )
        self._sync_client()
        result = self._client.post_or_fallback(
            prompt,
            fallback=None,
            options={"temperature": 0.7},
            timeout=60,
        )
        if result is None:
            logger.warning(
                "[Ollama] Script generation failed — returning original text"
            )
            return text
        return result.get("response", "").strip() or text

    def generate_topic_script(self, topic: str, language: str = "en") -> str:
        prompt = (
            f"You are a news reporter creating a video script. "
            f"Write a concise, informative script (3-5 short paragraphs) about: {topic}\n\n"
            f"Rules:\n"
            f"- Write in {language}\n"
            f"- Keep paragraphs short (1-3 sentences each)\n"
            f"- Use clear, conversational language suitable for text-to-speech\n"
            f"- Include specific facts, numbers, and key details\n"
            f"- Do NOT use [pause] or stage directions\n"
            f"- Do NOT use markdown or bullet points\n"
            f"- Separate paragraphs with blank lines\n"
            f"- End each sentence with a period\n\n"
            f"Script:"
        )
        self._sync_client()
        result = self._client.post_or_fallback(
            prompt,
            fallback=None,
            options={"temperature": 0.7, "num_predict": 1024},
            timeout=120,
        )
        if result is None:
            logger.warning("[Ollama] Topic script generation failed — using fallback")
            return os.getenv("OLLAMA_FALLBACK_SCRIPT", DEFAULT_FALLBACK_SCRIPT)
        script = result.get("response", "").strip()
        if script:
            logger.info(
                "[Ollama] Generated %d-char script for topic: '%s'", len(script), topic
            )
            return script
        return os.getenv("OLLAMA_FALLBACK_SCRIPT", DEFAULT_FALLBACK_SCRIPT)

    # ------------------------------------------------------------------
    # Model listing
    # ------------------------------------------------------------------

    @staticmethod
    def fetch_models_static(base_url: str) -> List[str]:
        if "/generate" in base_url:
            url = base_url.replace("/generate", "/tags")
        else:
            p = urlparse(base_url)
            url = f"{p.scheme}://{p.netloc}/api/tags"

        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            pass
        return []

    def get_available_models(self) -> List[str]:
        models = self.fetch_models_static(self.url)
        default_model = "gemma4:e2b"
        if default_model not in models:
            models.append(default_model)
        if self.model and self.model not in models:
            models.append(self.model)
        return models


class KeywordExtractor:
    """Orchestrates keyword extraction using Ollama and Spacy fallback"""

    def __init__(self, ollama_model: str = "gemma4:e2b"):
        self.ollama_extractor = OllamaKeywordExtractor(model=ollama_model)
        self.neuron_extractor = NeuronExtractor(model=ollama_model)

        try:
            self.nlp = spacy.load("en_core_web_md")
        except Exception:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                self.nlp = None
                logger.warning(
                    "[NLP] Local spaCy model not found. Using fallback methods."
                )

        self.relevant_pos = {"NOUN", "PROPN", "ADJ"}
        self.exclude_words = {
            "thing",
            "things",
            "something",
            "someone",
            "way",
            "time",
            "day",
            "year",
            "week",
            "month",
            "people",
            "person",
            "place",
            "lot",
            "intro",
            "outro",
            "welcome",
            "thanks",
            "watching",
            "subscribe",
        }
        self.used_keywords = set()
        # Stored spaCy embeddings for semantic duplicate avoidance
        self.used_embeddings: List[np.ndarray] = []
        self.semantic_threshold = float(os.getenv("SEMANTIC_DUP_THRESHOLD", "0.8"))

        # Entity handling + audit trail (new in this revision)
        self.entity_handler = EntityHandler()
        self.debug_mode = bool(
            os.getenv("KEYWORD_DEBUG_MODE", "False").lower() == "true"
        )
        self.selection_history: List[Dict] = []
        self._history_limit = int(os.getenv("KEYWORD_HISTORY_LIMIT", "200"))

        # Embedding backend for semantic duplicate detection. Defaults to the
        # existing spaCy vectors; "sentence_transformer" enables all-MiniLM-L6-v2
        # (opt-in via KEYWORD_EMBEDDING_MODEL — heavier, but context-aware).
        self.embedding_model = os.getenv("KEYWORD_EMBEDDING_MODEL", "spacy").lower()
        self._st_model = None

    def extract_keywords(
        self,
        text: str,
        top_n: int = 5,
        language: str = "en",
        use_neuron_ai: bool = True,
        use_snn: bool = False,
        theme: Optional[str] = None,
        entity: Optional[str] = None,
        entity_type: Optional[str] = None,
    ) -> List[str]:
        if not text.strip():
            return []

        entity_dict = (
            self.entity_handler.parse_entity(entity, entity_type) if entity else None
        )

        candidates = self._extract_spacy_local(text, top_n * 2)
        # Surface entity-derived keywords so the entity can be selected directly.
        if entity_dict:
            for ek in entity_dict["keywords"]:
                if ek not in candidates:
                    candidates.append(ek)

        if len(candidates) < 2 and language == "en":
            ollama_keywords = self.ollama_extractor.extract_keywords(
                text, min(4, top_n * 2), language, theme=theme
            )
            candidates.extend([k for k in ollama_keywords if k not in candidates])

        source = "spacy_local" if self.nlp else "empty"
        reasoning: List[str] = []
        selected: List[str] = []

        if candidates:
            if use_neuron_ai:
                # Augment the evaluation context with the global theme (and any
                # entity) so neural scoring is biased toward on-topic keywords.
                neuron_context = f"{text} Theme: {theme}" if theme else text
                if entity_dict:
                    neuron_context = f"{neuron_context} Entity: {entity_dict['raw']}"
                neuron_results = self.neuron_extractor.evaluate_keywords(
                    neuron_context, candidates, language, use_snn=use_snn
                )
                if neuron_results:
                    selected = [res["keyword"] for res in neuron_results[:top_n]]
                    source = "neuron_ai"
                    reasoning.append("Neural scoring selected top candidates")

            if not selected:
                ranked = self.rank_keywords(candidates)
                selected = ranked[:top_n]
                source = "heuristic_rank"
                reasoning.append("Heuristic category scoring selected candidates")

            # Entity-aware re-prioritisation (reorders, never invents keywords).
            if entity_dict:
                selected = self.entity_handler.rank_keywords_by_entity(
                    selected, entity_dict
                )
                reasoning.append(
                    f"Entity '{entity_dict['raw']}' ({entity_dict['type']}) "
                    f"prioritised matching keywords"
                )
        else:
            reasoning.append("No candidates extracted; returning empty")

        selected = selected[:top_n]
        self._record_audit(
            text, theme, entity_dict, candidates, selected, source, reasoning
        )
        return selected

    # ------------------------------------------------------------------
    # Audit trail
    # ------------------------------------------------------------------
    def _record_audit(
        self,
        text: str,
        theme: Optional[str],
        entity_dict: Optional[Dict],
        candidates: List[str],
        selected: List[str],
        source: str,
        reasoning: List[str],
    ) -> None:
        audit = {
            "timestamp": datetime.now().isoformat(),
            "context": {
                "text": (text[:120] + ("..." if len(text) > 120 else "")),
                "theme": theme,
            },
            "entity": entity_dict["raw"] if entity_dict else None,
            "entity_type": entity_dict["type"] if entity_dict else None,
            "candidates": candidates[:50],
            "selected": selected,
            "source": source,
            "reasoning": reasoning,
            "fallback_used": not selected,
        }
        self.selection_history.append(audit)
        if len(self.selection_history) > self._history_limit:
            self.selection_history = self.selection_history[-self._history_limit :]
        if self.debug_mode:
            logger.info("[KW-AUDIT] %s", audit)

    def set_debug_mode(self, enabled: bool) -> None:
        """Toggle verbose audit logging for keyword selection."""
        self.debug_mode = bool(enabled)

    def get_selection_history(self, limit: int = 100) -> List[Dict]:
        """Return recent selection decisions for debugging."""
        return self.selection_history[-limit:]

    def _extract_spacy_local(self, text: str, top_n: int = 5) -> List[str]:
        if not self.nlp:
            return []

        doc = self.nlp(text.lower())
        candidates = []

        for ent in doc.ents:
            if ent.label_ in {"GPE", "LOC", "ORG", "PRODUCT", "EVENT", "PERSON"}:
                candidates.append(ent.text)

        for chunk in doc.noun_chunks:
            clean_chunk = " ".join(
                [
                    t.text
                    for t in chunk
                    if not t.is_stop and t.pos_ in {"NOUN", "PROPN", "ADJ"}
                ]
            )
            if clean_chunk and len(clean_chunk.split()) >= 1:
                candidates.append(clean_chunk)

        for token in doc:
            if (
                token.pos_ in self.relevant_pos
                and not token.is_stop
                and len(token.text) > 2
                and token.text.isalpha()
                and token.text not in self.exclude_words
            ):
                candidates.append(token.text)

        if not candidates:
            return []

        counts = Counter(candidates)
        return [word for word, count in counts.most_common(top_n)]

    def _extract_spacy_fallback(self, text: str, top_n: int = 5) -> List[str]:
        spacy_url = os.getenv("SPACY_API_URL", "https://spacy.izdrail.com")
        try:
            pos_resp = requests.post(
                f"{spacy_url}/pos", json={"text": text.lower()}, timeout=10
            )
            candidates = []
            if pos_resp.status_code == 200:
                tokens = pos_resp.json()
                for token in tokens:
                    pos = token.get("pos")
                    word = token.get("text")
                    is_stop = token.get("is_stop", False)
                    if (
                        pos in self.relevant_pos
                        and not is_stop
                        and len(word) > 2
                        and word.isalpha()
                        and word not in self.exclude_words
                    ):
                        candidates.append(word)

            if not candidates:
                return []

            freq = Counter(candidates)
            return [word for word, count in freq.most_common(top_n)]
        except Exception as e:
            logger.warning("[NLP] Spacy fallback error: %s", e)
            return []

    def extract_theme(self, text: str, language: str = "en") -> Optional[str]:
        """Extract a global theme for the whole script (delegates to Ollama)."""
        return self.ollama_extractor.extract_theme(text, language)

    # ------------------------------------------------------------------
    # Semantic duplicate avoidance
    # ------------------------------------------------------------------

    def _embedding(self, keyword: str) -> Optional[np.ndarray]:
        """Return an embedding vector for a keyword, or None if unavailable.

        Uses sentence-transformers (all-MiniLM-L6-v2) when configured, otherwise
        the existing spaCy vectors. Both return a NumPy array so all downstream
        cosine-similarity code is unchanged.
        """
        if not keyword:
            return None
        if self.embedding_model == "sentence_transformer":
            return self._st_embedding(keyword)
        if not self.nlp:
            return None
        doc = self.nlp(keyword.lower())
        if not doc.vector_norm:
            return None
        return doc.vector

    def _st_embedding(self, keyword: str) -> Optional[np.ndarray]:
        """Lazily load and query a sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:  # pragma: no cover - optional dependency
            logger.warning(
                "[NLP] sentence_transformers unavailable (%s); using spaCy vectors",
                e,
            )
            self.embedding_model = "spacy"
            return self._embedding(keyword)
        if self._st_model is None:
            logger.info("[NLP] Loading sentence-transformers model all-MiniLM-L6-v2")
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
        vec = self._st_model.encode([keyword], normalize_embeddings=True)[0]
        return np.asarray(vec, dtype=np.float64)

    def is_semantically_unique(
        self, keyword: str, threshold: Optional[float] = None
    ) -> bool:
        """Return True if `keyword` is not too similar to already-used keywords.

        Uses spaCy word embeddings; falls back to exact-match when embeddings
        are unavailable (e.g. only the small spaCy model is installed).
        """
        if not keyword:
            return False
        # Cheap exact-match short-circuit
        if keyword in self.used_keywords:
            return False

        thr = threshold if threshold is not None else self.semantic_threshold
        emb = self._embedding(keyword)
        if emb is None or not self.used_embeddings:
            return True

        for used_emb in self.used_embeddings:
            norm = float(np.linalg.norm(emb) * np.linalg.norm(used_emb))
            if norm == 0.0:
                continue
            sim = float(np.dot(emb, used_emb) / norm)
            if sim > thr:
                return False
        return True

    def add_used_keyword(self, keyword: str) -> None:
        """Register a keyword as used (exact set + embedding for semantic checks)."""
        if not keyword:
            return
        self.used_keywords.add(keyword)
        emb = self._embedding(keyword)
        if emb is not None and emb.size:
            self.used_embeddings.append(emb)

    def get_best_unique_keyword(
        self, text: str, language: Optional[str] = None
    ) -> Optional[str]:
        keywords = self.extract_keywords(text, top_n=10)
        for kw in keywords:
            if kw not in self.used_keywords:
                self.add_used_keyword(kw)
                return kw
        return keywords[0] if keywords else None

    def clear_used(self):
        self.used_keywords.clear()
        self.used_embeddings.clear()
        self.neuron_extractor.clear_memory()

    # ------------------------------------------------------------------
    # SNN / neural sequence coherence
    # ------------------------------------------------------------------

    def _keyword_engagement(
        self, context: str, keyword: str, use_snn: bool = False
    ) -> float:
        """Neural engagement score for a (context, keyword) pair.

        Uses the cached local neuron signals; if `use_snn` is set and the
        BrainSimulator is available, the spiking-network biological score is
        used instead of the heuristic decision score.
        """
        signals = self.neuron_extractor._local_evaluate_signals(context, keyword)
        if not signals:
            return 0.0
        score = self.neuron_extractor._calculate_decision_score(signals)
        brain_sim = self.neuron_extractor.brain_simulator
        if use_snn and brain_sim is not None:
            try:
                snn_score, _ = brain_sim.evaluate_keyword_snn(context, keyword, signals)
                score = snn_score
            except Exception as e:
                logger.debug("[NLP] SNN engagement failed: %s", e)
        return float(score)

    def _keyword_similarity(self, a: str, b: str) -> float:
        ea, eb = self._embedding(a), self._embedding(b)
        if ea is None or eb is None:
            return 0.0
        norm = float(np.linalg.norm(ea) * np.linalg.norm(eb))
        return float(np.dot(ea, eb) / norm) if norm > 0 else 0.0

    def optimize_keyword_sequence(
        self,
        sentences: List[str],
        candidates_map: Dict[int, List[str]],
        theme: Optional[str] = None,
        use_snn: bool = False,
        coherence_weight: float = 0.5,
        engagement_weight: float = 1.0,
        beam_width: int = 4,
    ) -> Dict[int, Optional[str]]:
        """Pick one keyword per sentence maximising global narrative coherence.

        Uses beam search (width = ``beam_width``) over the candidate lists to
        avoid the local-maxima problem of the old greedy selection: each partial
        sequence is expanded by every candidate for the next sentence, scored on a
        blend of neural engagement and thematic smoothness (with a semantic-dup
        penalty), and only the top-``beam_width`` partials are retained per step.
        Returns {sentence_index: chosen_keyword}.
        """
        if not candidates_map:
            return {}

        beam_width = max(1, int(beam_width))

        # Precompute engagement once per (sentence, candidate) pair.
        engagement: Dict[Tuple[int, str], float] = {}
        for idx, cands in candidates_map.items():
            ctx = sentences[idx] if idx < len(sentences) else ""
            for kw in cands:
                engagement[(idx, kw)] = self._keyword_engagement(ctx, kw, use_snn)

        # Beam entry: (accumulated_score, chosen_keywords, prev_kw, emb_history)
        beam: List[
            Tuple[float, List[Optional[str]], Optional[str], List[Optional[np.ndarray]]]
        ] = [(0.0, [], None, [])]

        for idx in sorted(candidates_map.keys()):
            cands = candidates_map.get(idx, [])
            if not cands:
                beam = [(s, c + [None], p, h) for (s, c, p, h) in beam][:beam_width]
                continue

            expanded = []
            for score, seq, prev, emb_hist in beam:
                for kw in cands:
                    eng = engagement.get((idx, kw), 0.0)
                    coh = self._keyword_similarity(kw, prev) if prev else 0.0
                    dup_penalty = 0.0 if self._unique_against(kw, emb_hist) else 5.0
                    step = (
                        engagement_weight * eng + coherence_weight * coh - dup_penalty
                    )
                    kw_emb = self._embedding(kw)
                    expanded.append(
                        (
                            score + step,
                            seq + [kw],
                            kw,
                            emb_hist + ([kw_emb] if kw_emb is not None else []),
                        )
                    )

            if not expanded:
                continue
            expanded.sort(key=lambda x: x[0], reverse=True)
            beam = expanded[:beam_width]

        if not beam:
            return {i: None for i in candidates_map.keys()}

        # Prefer the longest complete sequence, then the highest score.
        best = max(beam, key=lambda x: (len(x[1]), x[0]))
        chosen_list = best[1]
        chosen: Dict[int, Optional[str]] = {}
        for i, idx in enumerate(sorted(candidates_map.keys())):
            kw = chosen_list[i] if i < len(chosen_list) else None
            chosen[idx] = kw
            if kw:
                self.add_used_keyword(kw)
        return chosen

    def _unique_against(self, kw: str, emb_history) -> bool:
        """Read-only uniqueness check against already-used keywords/embeddings
        plus a local ``emb_history`` (without mutating global state).

        Used by the beam search so partial sequences can be scored without
        permanently marking keywords as used until a final sequence is chosen.
        """
        if not kw:
            return False
        if kw in self.used_keywords:
            return False
        emb = self._embedding(kw)
        if emb is None:
            return True
        history = list(self.used_embeddings) + list(emb_history)
        for used_emb in history:
            norm = float(np.linalg.norm(emb) * np.linalg.norm(used_emb))
            if norm == 0.0:
                continue
            if float(np.dot(emb, used_emb) / norm) > self.semantic_threshold:
                return False
        return True

    def evaluate_keyword_sequence_coherence(
        self,
        sentences: List[str],
        keywords: List[Optional[str]],
        use_snn: bool = False,
    ) -> Tuple[float, List[Dict[str, float]]]:
        """Score the global coherence of an already-selected keyword sequence.

        Returns (coherence_score, per_keyword_details). Coherence rewards smooth
        thematic transitions between neighbours and penalizes engagement variance.
        """
        details: List[Dict[str, Any]] = []
        prev_emb = None
        coh_sum, n = 0.0, 0
        eng_values: List[float] = []
        for i, kw in enumerate(keywords):
            if not kw:
                details.append({"keyword": kw, "engagement": 0.0, "coherence": 0.0})
                continue
            ctx = sentences[i] if i < len(sentences) else ""
            eng = self._keyword_engagement(ctx, kw, use_snn)
            eng_values.append(eng)
            emb = self._embedding(kw)
            coh = 0.0
            if prev_emb is not None and emb is not None:
                norm = float(np.linalg.norm(emb) * np.linalg.norm(prev_emb))
                coh = float(np.dot(emb, prev_emb) / norm) if norm > 0 else 0.0
                coh_sum += coh
                n += 1
            details.append({"keyword": kw, "engagement": eng, "coherence": coh})
            prev_emb = emb

        avg_coh = coh_sum / n if n else 0.0
        eng_arr = np.array(eng_values) if eng_values else np.array([0.0])
        eng_std = float(np.std(eng_arr)) if eng_arr.size > 1 else 0.0
        coherence = float(np.clip(avg_coh - 0.3 * eng_std, 0.0, 1.0))
        return coherence, details

    def rank_keywords(self, keywords: List[str]) -> List[str]:
        stock_categories = {
            "nature",
            "forest",
            "ocean",
            "mountain",
            "sky",
            "sunset",
            "sunrise",
            "city",
            "cityscape",
            "building",
            "office",
            "street",
            "traffic",
            "people",
            "business",
            "meeting",
            "technology",
            "computer",
            "phone",
            "food",
            "cooking",
            "restaurant",
            "travel",
            "beach",
            "landscape",
            "water",
            "fire",
            "clouds",
            "rain",
            "snow",
            "night",
            "day",
            "abstract",
            "motion",
            "light",
            "color",
            "texture",
            "pattern",
            "hands",
            "work",
            "team",
            "collaboration",
            "innovation",
            "growth",
        }

        # Whole-word / phrase matching (word boundaries) so that substrings do
        # not falsely inflate scores, e.g. "birthday" must NOT match "day" and
        # "surrounding" must NOT match "running".
        def _word_match(phrase: str, text: str) -> bool:
            return bool(re.search(rf"\b{re.escape(phrase)}\b", text))

        def score_keyword(kw: str) -> float:
            score = 0.0
            kw_lower = kw.lower()

            if kw not in self.used_keywords:
                score += 10.0

            for category in stock_categories:
                if _word_match(category, kw_lower):
                    score += 5.0
                    break

            if len(kw.split()) == 2:
                score += 3.0

            generic_terms = {"thing", "stuff", "item", "object", "concept", "idea"}
            if kw_lower in generic_terms:
                score -= 5.0

            visual_actions = {
                "moving",
                "flowing",
                "growing",
                "building",
                "working",
                "walking",
                "running",
                "flying",
            }
            if any(_word_match(action, kw_lower) for action in visual_actions):
                score += 2.0

            return score

        ranked = sorted(keywords, key=score_keyword, reverse=True)
        return ranked

    def generate_fallback_keywords(self, keyword: str) -> List[str]:
        fallbacks = []
        kw_lower = keyword.lower()

        category_map = {
            "tree": ["forest", "nature", "woods"],
            "flower": ["garden", "nature", "plants"],
            "garden": ["nature", "plants", "outdoor"],
            "ocean": ["water", "sea", "waves"],
            "mountain": ["landscape", "nature", "outdoor"],
            "river": ["water", "nature", "stream"],
            "building": ["city", "architecture", "urban"],
            "office": ["business", "workplace", "indoor"],
            "street": ["city", "urban", "traffic"],
            "car": ["traffic", "transportation", "vehicle"],
            "computer": ["technology", "office", "digital"],
            "phone": ["technology", "communication", "mobile"],
            "code": ["technology", "programming", "digital"],
            "data": ["technology", "digital", "abstract"],
            "meeting": ["business", "people", "office"],
            "team": ["business", "people", "collaboration"],
            "work": ["business", "office", "people"],
            "cooking": ["food", "kitchen", "chef"],
        }

        for key, values in category_map.items():
            if key in kw_lower:
                fallbacks.extend(values)
                break

        words = keyword.split()
        if len(words) > 1:
            fallbacks.append(words[0])

        generic_fallbacks = [
            "abstract",
            "motion",
            "light",
            "texture",
            "landscape",
            "cityscape",
        ]

        unique_fallbacks = []
        seen = set([kw_lower])

        for fb in fallbacks + generic_fallbacks:
            if fb not in seen:
                unique_fallbacks.append(fb)
                seen.add(fb)
                if len(unique_fallbacks) >= 5:
                    break

        return unique_fallbacks

    def sanitize_keyword(self, keyword: str) -> str:
        if not keyword:
            return ""
        kw = re.sub(r"[\*\-•\n]+", "", keyword)
        return re.sub(r"\s+", " ", kw).strip().lower()

    def enrich_keyword_context(
        self,
        keyword: str,
        context: Optional[str],
        max_words: int = 3,
        enabled: Optional[bool] = None,
    ) -> str:
        """Disambiguate a polysemous keyword by appending salient context words.

        e.g. ``enrich_keyword_context("bank", "a river bank at sunset")`` ->
        ``"bank river"``. The enriched query is only used for the stock-footage
        search string (selection/dedup still use the raw keyword). Disabled when
        ``KEYWORD_CONTEXT_ENRICH=false``; falls back to a regex content-word
        extractor when spaCy is unavailable.
        """
        if enabled is None:
            enabled = os.getenv("KEYWORD_CONTEXT_ENRICH", "true").lower() != "false"
        if not enabled or not context or not keyword:
            return keyword

        kw_lower = keyword.lower()
        extra: List[str] = []

        if self.nlp:
            doc = self.nlp(context.lower())
            for tok in doc:
                if tok.is_stop or tok.is_punct:
                    continue
                if tok.text == kw_lower or tok.text in kw_lower.split():
                    continue
                if len(tok.text) <= 2 or not tok.text.isalpha():
                    continue
                if tok.pos_ in {"NOUN", "PROPN", "VERB", "ADJ"}:
                    extra.append(tok.text)
                if len(extra) >= max_words:
                    break
        else:
            stop = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "of",
                "to",
                "in",
                "on",
                "at",
                "for",
                "with",
                "is",
                "are",
                "was",
                "were",
                "this",
                "that",
                "it",
                "as",
                "by",
                "from",
                "be",
                "we",
                "you",
                "they",
            }
            for w in re.findall(r"[a-zA-Z]+", context.lower()):
                if w == kw_lower or w in stop or len(w) <= 2:
                    continue
                extra.append(w)
                if len(extra) >= max_words:
                    break

        if extra:
            return f"{keyword} {' '.join(extra)}"
        return keyword

    def get_available_models(self) -> List[str]:
        return self.ollama_extractor.get_available_models()

    @property
    def api_url(self) -> str:
        return self.ollama_extractor.url

    @api_url.setter
    def api_url(self, value: str):
        if value and value != self.ollama_extractor.url:
            logger.info("[Ollama] Updating API URL to: %s", value)
            self.ollama_extractor.url = value
            self.ollama_extractor._client.set_url(value)
            self.ollama_extractor.cache.clear()

    @property
    def model(self) -> str:
        return self.ollama_extractor.model

    @model.setter
    def model(self, value: str):
        self.ollama_extractor.model = value
        self.ollama_extractor._client.set_model(value)

    def generate_social_media_descriptions(
        self, text: str, keywords: List[str], language: str = "en"
    ) -> str:
        return self.ollama_extractor.generate_social_media_descriptions(
            text, keywords, language
        )

    def generate_script_from_text(self, text: str) -> str:
        return self.ollama_extractor.generate_script_from_text(text)

    def generate_topic_script(self, topic: str, language: str = "en") -> str:
        return self.ollama_extractor.generate_topic_script(topic, language)

    def extract_mood_keyword(self, text: str) -> str:
        mood = self.ollama_extractor.extract_mood_keyword(text)
        if mood:
            return mood

        text_lower = text.lower()
        if any(
            w in text_lower
            for w in ["war", "battle", "fight", "epic", "victory", "strong"]
        ):
            return "Epic"
        if any(w in text_lower for w in ["peace", "relax", "ocean", "sleep", "calm"]):
            return "Relaxing"
        if any(w in text_lower for w in ["happy", "fun", "upbeat", "party"]):
            return "Happy"
        if any(w in text_lower for w in ["sad", "dark", "alone", "scary"]):
            return "Dark"

        return "Cinematic"
