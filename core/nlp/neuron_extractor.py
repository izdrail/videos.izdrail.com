import json
import logging
import os
import re
import sqlite3
import hashlib
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import spacy

from .ollama_client import OllamaClient

try:
    from .brain_simulator import BrainSimulator

    SNN_AVAILABLE = True
except ImportError:
    SNN_AVAILABLE = False

logger = logging.getLogger(__name__)


class NeuronExtractor:
    """
    Implements a 'Neuron AI' decision architecture for keyword evaluation.
    Maps brain regions to specific AI evaluation roles.
    """

    def __init__(self, model: str = "gemma4:e2b", url: Optional[str] = None):
        self.model = model
        self.url = url or os.getenv(
            "OLLAMA_API_URL", "https://ai.izdrail.com/api/generate"
        )
        self.brain_simulator = BrainSimulator() if SNN_AVAILABLE else None

        # Build resilient client
        self._client = OllamaClient(
            model=self.model,
            url=self.url,
            max_retries=int(os.getenv("OLLAMA_MAX_RETRIES", "3")),
            base_delay=float(os.getenv("OLLAMA_RETRY_BASE_DELAY", "1.0")),
            timeout=int(os.getenv("OLLAMA_TIMEOUT", "60")),
            cache_max_size=int(os.getenv("OLLAMA_CACHE_MAX_SIZE", "512")),
        )

        # Local NLP for vector similarity
        try:
            self.nlp = spacy.load("en_core_web_md")
        except Exception:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                self.nlp = None

        # Anchor concepts for neuron mapping
        self.anchors = {
            "att": "shocking surprise vibrant attention focus hook",
            "amy": "emotion exciting dangerous intense vivid",
            "rew": "pleasure reward success desire wealth dopamine luxury",
            "ins": "boring painful waste irrelevant ugly clinical disgusting",
            "vmp": "value worth quality premium logic beneficial",
            "hip": "familiar memory consistent common traditional",
            "dlp": "authority logic professional official expensive",
        }
        self._anchor_docs = None
        if self.nlp:
            self._anchor_docs = {k: self.nlp(v) for k, v in self.anchors.items()}

        # ── Neural evaluation cache (in-memory + optional SQLite) ──
        self._eval_cache: Dict[str, Any] = {}
        self._eval_cache_max = int(os.getenv("NEURON_CACHE_MAX", "2000"))
        # Defaults to a local SQLite file so neural evaluations persist across
        # runs (set NEURON_CACHE_DB= (empty) or a path to override/disable).
        self._cache_db_path = os.getenv("NEURON_CACHE_DB", "neural_cache.db") or None
        self._cache_db = None
        if self._cache_db_path:
            try:
                self._cache_db = sqlite3.connect(
                    self._cache_db_path, check_same_thread=False
                )
                self._cache_db.execute(
                    "CREATE TABLE IF NOT EXISTS neuron_signals ("
                    "key TEXT PRIMARY KEY, value TEXT)"
                )
                self._cache_db.commit()
            except Exception as e:
                logger.warning("[NeuronAI] Neural cache DB unavailable: %s", e)
                self._cache_db = None

    def _signal_cache_key(self, context: str, target: str) -> str:
        payload = f"{self.model}|{context[:300]}|{target}"
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def _signal_cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self._eval_cache:
            return self._eval_cache[key]
        if self._cache_db is not None:
            try:
                row = self._cache_db.execute(
                    "SELECT value FROM neuron_signals WHERE key=?", (key,)
                ).fetchone()
                if row:
                    return json.loads(row[0])
            except Exception:
                pass
        return None

    def _signal_cache_set(self, key: str, value: Dict[str, Any]) -> None:
        self._eval_cache[key] = value
        if len(self._eval_cache) > self._eval_cache_max:
            # Simple FIFO eviction
            try:
                self._eval_cache.pop(next(iter(self._eval_cache)))
            except Exception:
                pass
        if self._cache_db is not None:
            try:
                self._cache_db.execute(
                    "INSERT OR REPLACE INTO neuron_signals (key, value) VALUES (?, ?)",
                    (key, json.dumps(value)),
                )
                self._cache_db.commit()
            except Exception:
                pass

    def _sync_client(self) -> None:
        self._client.set_model(self.model)
        self._client.set_url(self.url)

    def evaluate_keywords(
        self,
        text: str,
        candidates: List[str],
        language: str = "en",
        use_snn: bool = False,
    ) -> List[Dict[str, Any]]:
        candidates = candidates[:10]

        logger.debug(
            "[NeuronAI] Local evaluating %d keywords for: %s...",
            len(candidates),
            text[:30],
        )
        results = []
        for kw in candidates:
            signals = self._local_evaluate_signals(text, kw)
            if signals:
                signals["keyword"] = kw
                results.append(signals)

        if not results:
            results = self._query_neurons_batch(text, candidates, language)

        for res in results:
            if use_snn and self.brain_simulator:
                score, details = self.brain_simulator.evaluate_keyword_snn(
                    text, res.get("keyword", ""), res
                )
                res["decision_score"] = score
                res["snn_details"] = details
            else:
                res["decision_score"] = self._calculate_decision_score(res)

        # Incorporate CLIP similarity scoring for visual semantic relevance
        try:
            from core.visual.clip_scorer import CLIPScorer
            clip_scorer = CLIPScorer()
            for res in results:
                media = res.get("media", {})
                img_src = (
                    media.get("thumbnail")
                    or media.get("url")
                    or media.get("path")
                )
                if img_src:
                    clip_score = clip_scorer.compute_similarity(text, img_src)
                    res["clip_score"] = clip_score
                    # Blend CLIP visual relevance score with decision_score
                    res["decision_score"] = res.get("decision_score", 0.5) * 0.6 + clip_score * 0.4
        except Exception as e:
            logger.debug("[NeuronAI] CLIP scoring skipped: %s", e)

        results.sort(key=lambda x: x.get("decision_score", 0), reverse=True)
        return results

    def evaluate_media(
        self,
        text: str,
        media_candidates: List[Dict[str, Any]],
        language: str = "en",
        use_snn: bool = False,
    ) -> List[Dict[str, Any]]:
        candidates = media_candidates[:5]

        logger.debug("[NeuronAI] Local evaluating %d media items...", len(candidates))
        results = []
        for i, media in enumerate(candidates):
            media_info = f"{media.get('title', '')} {media.get('tags', '')}"
            signals = self._local_evaluate_signals(text, media_info)
            if signals:
                signals["index"] = i
                signals["media"] = media
                results.append(signals)

        if not results:
            results = self._query_media_batch(text, candidates, language)

        for res in results:
            if use_snn and self.brain_simulator:
                score, details = self.brain_simulator.evaluate_keyword_snn(
                    text, res.get("media", {}).get("title", ""), res
                )
                res["decision_score"] = score
                res["snn_details"] = details
            else:
                res["decision_score"] = self._calculate_decision_score(res)

        results.sort(key=lambda x: x.get("decision_score", 0), reverse=True)
        return results

    def _local_evaluate_signals(
        self, context: str, target: str
    ) -> Optional[Dict[str, Any]]:
        cache_key = self._signal_cache_key(context, target)
        cached = self._signal_cache_get(cache_key)
        if cached is not None:
            return cached

        signals = self._compute_signals(context, target)
        if signals is not None:
            self._signal_cache_set(cache_key, signals)
        return signals

    def _compute_signals(self, context: str, target: str) -> Optional[Dict[str, Any]]:
        if not self.nlp or not self._anchor_docs:
            return None

        target_doc = self.nlp(target.lower())
        if not target_doc.vector_norm:
            return None

        context_doc_full = self.nlp(context[:500].lower())
        essence_tokens = []
        for t in context_doc_full:
            if t.is_stop or t.is_punct:
                continue
            if t.pos_ in {"NOUN", "PROPN"}:
                essence_tokens.extend([t.text, t.text])
            elif t.pos_ in {"ADJ", "VERB"}:
                essence_tokens.append(t.text)

        if not essence_tokens:
            essence_tokens = [t.text for t in context_doc_full if not t.is_stop]

        context_clean = " ".join(essence_tokens[:40])
        context_doc = self.nlp(context_clean)

        def get_sim(anchor_key):
            sim = target_doc.similarity(self._anchor_docs[anchor_key])
            return float(np.clip((sim - 0.2) / 0.6, 0.0, 1.0))

        return {
            "attention": get_sim("att"),
            "amygdala": {"salience": get_sim("amy")},
            "reward": {"dopamine": get_sim("rew")},
            "insula": {"pain": get_sim("ins")},
            "vmpfc": {"value": get_sim("vmp")},
            "hippocampus": {
                "consistency": float(
                    np.clip(target_doc.similarity(context_doc), 0.0, 1.0)
                )
            },
            "dlpfc": {"authority": get_sim("dlp")},
        }

    def _query_neurons(
        self, context: str, target: str, language: str
    ) -> Optional[Dict[str, Any]]:
        prompt = f"""
        Evaluate the following {target} in the context of this script segment: "{context[:500]}..."

        Act as a Neuromarketing Decision Architect. Evaluate {target} in context: "{context[:300]}..."

        Target neurons:
        1. Attention, 2. Amygdala (salience), 3. Reward (desire), 4. Insula (pain), 5. vmPFC (value), 6. Hippocampus, 7. dlPFC.

        Rules:
        - Each neuron emits a numerical signal between 0.0 and 1.0.
        - Return ONLY a valid JSON object.
        - The JSON must follow this structure EXACTLY:
        {{
          "attention": 0.0,
          "amygdala": {{ "emotion": "name", "salience": 0.0 }},
          "reward": {{ "desire": 0.0, "dopamine": 0.0 }},
          "insula": {{ "pain": 0.0, "discomfort": 0.0 }},
          "vmpfc": {{ "value": 0.0, "worth_it": true }},
          "hippocampus": {{ "consistency": 0.0 }},
          "dlpfc": {{ "authority": 0.0, "override": false }}
        }}
        """
        self._sync_client()
        result = self._client.post_or_fallback(
            prompt,
            fallback=None,
            fmt="json",
            options={"temperature": 0.2, "num_predict": 256},
            timeout=60,
        )
        if result is None:
            return None

        raw = result.get("response", "").strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                return None
        return None

    def _query_neurons_batch(
        self, context: str, keywords: List[str], language: str
    ) -> List[Dict[str, Any]]:
        kw_list = ", ".join([f'"{k}"' for k in keywords])
        prompt = f"""
        Act as a Neuromarketing Decision Architecture and evaluate the following keywords in the context of: "{context[:500]}..."

        Keywords to evaluate: [{kw_list}]

        For EACH keyword, provide emission signals for these neurons:
        1. Attention Scanner, 2. Amygdala (salience), 3. Ventral Striatum (reward/desire/dopamine),
        4. Insula (pain/discomfort), 5. vmPFC (value), 6. Hippocampus (consistency), 7. dlPFC (authority).

        Rules:
        - Signals 0.0 to 1.0.
        - Return ONLY JSON: {{ "evals": [{{ "kw": "name", "att": 0.0, "amy": 0.0, "rew": 0.0, "ins": 0.0, "vmp": 0.0, "hip": 0.0, "dlp": 0.0 }}, ...] }}
        """
        self._sync_client()
        result = self._client.post_or_fallback(
            prompt,
            fallback=None,
            fmt="json",
            timeout=180,
        )
        if result is None:
            return []

        data = result.get("response", "{}")
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return []

        evs = data.get("evals", [])
        results = []
        for e in evs:
            results.append(
                {
                    "keyword": e.get("kw"),
                    "attention": e.get("att", 0.5),
                    "amygdala": {"salience": e.get("amy", 0.0)},
                    "reward": {"dopamine": e.get("rew", 0.0)},
                    "insula": {"pain": e.get("ins", 0.0)},
                    "vmpfc": {"value": e.get("vmp", 0.0)},
                    "hippocampus": {"consistency": e.get("hip", 0.0)},
                    "dlpfc": {"authority": e.get("dlp", 0.0)},
                }
            )
        return results

    def _query_media_batch(
        self, context: str, media_items: List[Dict[str, Any]], language: str
    ) -> List[Dict[str, Any]]:
        items_summary = []
        for i, m in enumerate(media_items):
            summ = f"ID={i}: Title={m.get('title', 'Untitled')}, Tags={str(m.get('tags', ''))[:50]}"
            items_summary.append(summ)

        items_str = "\n".join(items_summary)
        prompt = f"""
        Evaluate these media items for the context: "{context[:500]}..."

        Items:
        {items_str}

        Generate 0.0-1.0 signals for: att, amy, rew, ins, vmp, hip, dlp.
        Return ONLY JSON object: {{ "evals": [{{ "idx": 0, "att": 0.0, "amy": 0.0, "rew": 0.0, "ins": 0.0, "vmp": 0.0, "hip": 0.0, "dlp": 0.0 }}, ...] }}
        """
        self._sync_client()
        result = self._client.post_or_fallback(
            prompt,
            fallback=None,
            fmt="json",
            timeout=180,
        )
        if result is None:
            return []

        data = result.get("response", "{}")
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return []

        evs = data.get("evals", [])
        results = []
        for e in evs:
            idx = e.get("idx")
            if idx is not None and idx < len(media_items):
                results.append(
                    {
                        "index": idx,
                        "media": media_items[idx],
                        "attention": e.get("att", 0.5),
                        "amygdala": {"salience": e.get("amy", 0.0)},
                        "reward": {"dopamine": e.get("rew", 0.0)},
                        "insula": {"pain": e.get("ins", 0.0)},
                        "vmpfc": {"value": e.get("vmp", 0.0)},
                        "hippocampus": {"consistency": e.get("hip", 0.0)},
                        "dlpfc": {"authority": e.get("dlp", 0.0)},
                    }
                )
        return results

    def _calculate_decision_score(self, neuron_output: Dict[str, Any]) -> float:
        try:
            pain = neuron_output.get("insula", {}).get("pain", 0.0)
            if pain > 0.7:
                return -1.0

            reward = neuron_output.get("reward", {}).get("dopamine", 0.0) * 1.5
            emotion = neuron_output.get("amygdala", {}).get("salience", 0.0) * 1.2

            value = neuron_output.get("vmpfc", {}).get("value", 0.0) * 1.0

            identity = neuron_output.get("dlpfc", {}).get("authority", 0.0)
            if neuron_output.get("dlpfc", {}).get("override", False):
                identity += 0.5

            attention = neuron_output.get("attention", 0.5) + 0.5

            consistency = neuron_output.get("hippocampus", {}).get("consistency", 0.5)

            bad_content_penalty = 0.0
            media_title = neuron_output.get("media", {}).get("title", "").lower()
            bad_keywords = [
                "lyrics",
                "official video",
                "music video",
                "interview",
                "commentary",
                "vlog",
                "podcast",
                "review",
                "teaser",
                "trailer",
            ]
            for bk in bad_keywords:
                if bk in media_title:
                    bad_content_penalty += 0.4

            if "lyrics" in media_title:
                bad_content_penalty += 1.0

            score = (
                (reward + emotion + value + identity)
                + (consistency * 5.0)
                - (pain * 2.0)
                - (bad_content_penalty * 3.0)
            )
            return score * attention

        except (KeyError, TypeError) as e:
            logger.warning("[NeuronAI] Error calculating score: %s", e)
            return 0.0

    def clear_memory(self):
        if self.brain_simulator:
            self.brain_simulator.clear_memory()
