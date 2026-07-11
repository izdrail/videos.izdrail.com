"""
Keyword Extraction
Uses AI and NLP to extract visual keywords from text
"""

import logging
import os
import re
from collections import Counter
from typing import List, Optional
from urllib.parse import urlparse

import requests
import spacy

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

    def __init__(self, model: str = "mistral:7b", url: Optional[str] = None):
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
        self, text: str, top_n: int = 5, language: str = "en"
    ) -> List[str]:
        cache_key = f"{text[:100]}_{top_n}_{language}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = (
            f"You are a stock footage search expert. Extract {top_n} visual keywords from the text below. "
            f"Rules:\n"
            f"- Each keyword must be 1-2 words maximum\n"
            f"- Prioritize: physical objects, locations, actions, nature, technology, people activities\n"
            f"- Use common stock footage terms (e.g., 'city skyline', 'ocean waves', 'forest', 'office')\n"
            f"- Avoid abstract concepts unless they have clear visual representations\n"
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
        default_model = "mistral:7b"
        if default_model not in models:
            models.append(default_model)
        if self.model and self.model not in models:
            models.append(self.model)
        return models


class KeywordExtractor:
    """Orchestrates keyword extraction using Ollama and Spacy fallback"""

    def __init__(self, ollama_model: str = "mistral:7b"):
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

    def extract_keywords(
        self,
        text: str,
        top_n: int = 5,
        language: str = "en",
        use_neuron_ai: bool = True,
        use_snn: bool = False,
    ) -> List[str]:
        if not text.strip():
            return []

        candidates = self._extract_spacy_local(text, top_n * 2)

        if len(candidates) < 2 and language == "en":
            ollama_keywords = self.ollama_extractor.extract_keywords(
                text, min(4, top_n * 2), language
            )
            candidates.extend([k for k in ollama_keywords if k not in candidates])

        if candidates:
            if use_neuron_ai:
                neuron_results = self.neuron_extractor.evaluate_keywords(
                    text, candidates, language, use_snn=use_snn
                )
                if neuron_results:
                    return [res["keyword"] for res in neuron_results[:top_n]]

            ranked = self.rank_keywords(candidates)
            return ranked[:top_n]

        return []

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

    def get_best_unique_keyword(
        self, text: str, language: Optional[str] = None
    ) -> Optional[str]:
        keywords = self.extract_keywords(text, top_n=10)
        for kw in keywords:
            if kw not in self.used_keywords:
                self.used_keywords.add(kw)
                return kw
        return keywords[0] if keywords else None

    def clear_used(self):
        self.used_keywords.clear()
        self.neuron_extractor.clear_memory()

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

        def score_keyword(kw: str) -> float:
            score = 0.0
            kw_lower = kw.lower()

            if kw not in self.used_keywords:
                score += 10.0

            for category in stock_categories:
                if category in kw_lower:
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
            if any(action in kw_lower for action in visual_actions):
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
