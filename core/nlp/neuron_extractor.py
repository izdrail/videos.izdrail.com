import os
import json
import requests
import re
from typing import List, Dict, Any, Optional
try:
    from .brain_simulator import BrainSimulator
    SNN_AVAILABLE = True
except ImportError:
    SNN_AVAILABLE = False
import spacy
import numpy as np

class NeuronExtractor:
    """
    Implements a 'Neuron AI' decision architecture for keyword evaluation.
    Maps brain regions to specific AI evaluation roles.
    """
    
    def __init__(self, model: str = "mistral:7b", url: Optional[str] = None):
        self.model = model
        self.url = url or os.getenv("OLLAMA_API_URL", "https://ai.izdrail.com/api/generate")
        self.brain_simulator = BrainSimulator() if SNN_AVAILABLE else None
        
        # Local NLP for vector similarity
        try:
            self.nlp = spacy.load("en_core_web_md")
        except:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                self.nlp = None
                
        # Anchor concepts for neuron mapping
        self.anchors = {
            "att": "shocking surprise vibrant attention focus hook",
            "amy": "emotion exciting dangerous intense vivid",
            "rew": "pleasure reward success desire wealth dopamine luxury",
            "ins": "boring painful waste irrelevant ugly clinical disgusting",
            "vmp": "value worth quality premium logic beneficial",
            "hip": "familiar memory consistent common traditional",
            "dlp": "authority logic professional official expensive"
        }
        self._anchor_docs = None
        if self.nlp:
            self._anchor_docs = {k: self.nlp(v) for k, v in self.anchors.items()}
        
    def evaluate_keywords(self, text: str, candidates: List[str], language: str = 'en', use_snn: bool = False) -> List[Dict[str, Any]]:
        """
        Evaluates a list of candidate keywords using the Neuron AI architecture.
        Uses batching to reduce API calls and improve speed.
        """
        # Limit candidates to a reasonable number
        candidates = candidates[:10]
        
        # USE LOCAL EVALUATION BY DEFAULT FOR SPEED
        print(f"[NeuronAI] Local evaluating {len(candidates)} keywords for: {text[:30]}...")
        results = []
        for kw in candidates:
            signals = self._local_evaluate_signals(text, kw)
            if signals:
                signals['keyword'] = kw
                results.append(signals)
        
        if not results:
             # Deep fallback to Ollama if local fails
             results = self._query_neurons_batch(text, candidates, language)

        # Calculate final decision scores (SNN or static)
        for res in results:
            if use_snn and self.brain_simulator:
                score, details = self.brain_simulator.evaluate_keyword_snn(text, res.get('keyword', ''), res)
                res['decision_score'] = score
                res['snn_details'] = details
            else:
                res['decision_score'] = self._calculate_decision_score(res)
                
        results.sort(key=lambda x: x.get('decision_score', 0), reverse=True)
        return results

    def evaluate_media(self, text: str, media_candidates: List[Dict[str, Any]], language: str = 'en', use_snn: bool = False) -> List[Dict[str, Any]]:
        """
        Evaluates a list of candidate media items (videos/images) using the Neuron AI architecture.
        Uses batching for performance.
        """
        # We only evaluate the first 5 candidates to save on API calls/time
        candidates = media_candidates[:5]
        
        print(f"[NeuronAI] Local evaluating {len(candidates)} media items...")
        results = []
        for i, media in enumerate(candidates):
            media_info = f"{media.get('title', '')} {media.get('tags', '')}"
            signals = self._local_evaluate_signals(text, media_info)
            if signals:
                signals['index'] = i
                signals['media'] = media
                results.append(signals)
        
        if not results:
            results = self._query_media_batch(text, candidates, language)

        # Calculate final decision scores
        for res in results:
            if use_snn and self.brain_simulator:
                score, details = self.brain_simulator.evaluate_keyword_snn(text, res.get('media', {}).get('title', ''), res)
                res['decision_score'] = score
                res['snn_details'] = details
            else:
                res['decision_score'] = self._calculate_decision_score(res)
                
        results.sort(key=lambda x: x.get('decision_score', 0), reverse=True)
        return results

    def _local_evaluate_signals(self, context: str, target: str) -> Optional[Dict[str, Any]]:
        """Uses local spaCy vectors to estimate neuron signals without Ollama."""
        if not self.nlp or not self._anchor_docs:
            return None
            
        target_doc = self.nlp(target.lower())
        if not target_doc.vector_norm:
            return None
            
        # Refine context: Focus on significant tokens (Nouns, Entities, Adjectives)
        # Nouns/Proper Nouns are weighted twice (Subject-Booster)
        context_doc_full = self.nlp(context[:500].lower())
        essence_tokens = []
        for t in context_doc_full:
            if t.is_stop or t.is_punct: continue
            if t.pos_ in {"NOUN", "PROPN"}:
                essence_tokens.extend([t.text, t.text]) # Double weight
            elif t.pos_ in {"ADJ", "VERB"}:
                essence_tokens.append(t.text)
                
        if not essence_tokens:
            essence_tokens = [t.text for t in context_doc_full if not t.is_stop]
            
        context_clean = " ".join(essence_tokens[:40]) 
        context_doc = self.nlp(context_clean)
        
        def get_sim(anchor_key):
            sim = target_doc.similarity(self._anchor_docs[anchor_key])
            # Normalize sim (usually 0.3-0.8 range for md) to 0.0-1.0
            return float(np.clip((sim - 0.2) / 0.6, 0.0, 1.0))

        return {
            "attention": get_sim("att"),
            "amygdala": {"salience": get_sim("amy")},
            "reward": {"dopamine": get_sim("rew")},
            "insula": {"pain": get_sim("ins")},
            "vmpfc": {"value": get_sim("vmp")},
            "hippocampus": {"consistency": float(np.clip(target_doc.similarity(context_doc), 0.0, 1.0))},
            "dlpfc": {"authority": get_sim("dlp")}
        }

    def _query_neurons(self, context: str, target: str, language: str) -> Optional[Dict[str, Any]]:
        """
        Queries Ollama to get the 'neuron signals' for a specific target (keyword or media) in context.
        """
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
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2, "num_predict": 256}
        }
        
        try:
            response = requests.post(self.url, json=payload, timeout=60)
            if response.status_code == 200:
                raw = response.json().get("response", "").strip()
                # Basic cleanup in case of extra text
                json_match = re.search(r'\{.*\}', raw, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
        except Exception as e:
            print(f"[NeuronAI] Error evaluating target '{target}': {e}")
            
        return None

    def _query_neurons_batch(self, context: str, keywords: List[str], language: str) -> List[Dict[str, Any]]:
        """Queries Ollama to evaluate multiple keywords at once."""
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
        try:
            response = requests.post(
                self.url,
                json={"model": self.model, "prompt": prompt, "stream": False, "format": "json"},
                timeout=180
            )
            if response.status_code == 200:
                data = response.json().get('response', '{}')
                if isinstance(data, str): data = json.loads(data)
                evs = data.get("evals", [])
                # Map flat signals back to our structure
                results = []
                for e in evs:
                    results.append({
                        "keyword": e.get("kw"),
                        "attention": e.get("att", 0.5),
                        "amygdala": {"salience": e.get("amy", 0.0)},
                        "reward": {"dopamine": e.get("rew", 0.0)},
                        "insula": {"pain": e.get("ins", 0.0)},
                        "vmpfc": {"value": e.get("vmp", 0.0)},
                        "hippocampus": {"consistency": e.get("hip", 0.0)},
                        "dlpfc": {"authority": e.get("dlp", 0.0)}
                    })
                return results
        except Exception as e:
            print(f"[NeuronAI] Batch query error: {e}")
        return []

    def _query_media_batch(self, context: str, media_items: List[Dict[str, Any]], language: str) -> List[Dict[str, Any]]:
        """Queries Ollama to evaluate multiple media items at once."""
        items_summary = []
        for i, m in enumerate(media_items):
            summ = f"ID={i}: Title={m.get('title','Untitled')}, Tags={str(m.get('tags',''))[:50]}"
            items_summary.append(summ)
            
        items_str = "\n".join(items_summary)
        prompt = f"""
        Evaluate these media items for the context: "{context[:500]}..."
        
        Items:
        {items_str}
        
        Generate 0.0-1.0 signals for: att, amy, rew, ins, vmp, hip, dlp.
        Return ONLY JSON object: {{ "evals": [{{ "idx": 0, "att": 0.0, "amy": 0.0, "rew": 0.0, "ins": 0.0, "vmp": 0.0, "hip": 0.0, "dlp": 0.0 }}, ...] }}
        """
        try:
            response = requests.post(
                self.url,
                json={"model": self.model, "prompt": prompt, "stream": False, "format": "json"},
                timeout=180
            )
            if response.status_code == 200:
                data = response.json().get('response', '{}')
                if isinstance(data, str): data = json.loads(data)
                evs = data.get("evals", [])
                results = []
                for e in evs:
                    idx = e.get("idx")
                    if idx is not None and idx < len(media_items):
                        results.append({
                            "index": idx,
                            "media": media_items[idx], # CRITICAL: Re-attach original media object
                            "attention": e.get("att", 0.5),
                            "amygdala": {"salience": e.get("amy", 0.0)},
                            "reward": {"dopamine": e.get("rew", 0.0)},
                            "insula": {"pain": e.get("ins", 0.0)},
                            "vmpfc": {"value": e.get("vmp", 0.0)},
                            "hippocampus": {"consistency": e.get("hip", 0.0)},
                            "dlpfc": {"authority": e.get("dlp", 0.0)}
                        })
                return results
        except Exception as e:
            print(f"[NeuronAI] Media batch query error: {e}")
        return []

    def _calculate_decision_score(self, neuron_output: Dict[str, Any]) -> float:
        """
        Calculates a final decision score based on the neuron signals.
        Order of biological importance: Emotion/Reward > Pain > Logic/Identity.
        """
        try:
            # High pain can abort/kill the score
            pain = neuron_output.get('insula', {}).get('pain', 0.0)
            if pain > 0.7:
                return -1.0 # Significant pain override
                
            # Reward and Emotion (Base drive)
            reward = neuron_output.get('reward', {}).get('dopamine', 0.0) * 1.5
            emotion = neuron_output.get('amygdala', {}).get('salience', 0.0) * 1.2
            
            # Rational Value
            value = neuron_output.get('vmpfc', {}).get('value', 0.0) * 1.0
            
            # Identity / Logic Override
            identity = neuron_output.get('dlpfc', {}).get('authority', 0.0)
            if neuron_output.get('dlpfc', {}).get('override', False):
                identity += 0.5
                
            # Attention (Multiplier for visibility)
            attention = neuron_output.get('attention', 0.5) + 0.5 # Range 0.5 to 1.5
            
            # Relevance / Consistency (Hippocampus) - CRITICAL for user context
            consistency = neuron_output.get('hippocampus', {}).get('consistency', 0.5)
            
            # CONTENT PENALTY SYSTEM
            # Penalize talking heads, lyrics, or specific YouTube commentary styles
            bad_content_penalty = 0.0
            media_title = neuron_output.get('media', {}).get('title', '').lower()
            bad_keywords = ["lyrics", "official video", "music video", "interview", "commentary", "vlog", "podcast", "review", "teaser", "trailer"]
            for bk in bad_keywords:
                if bk in media_title:
                    bad_content_penalty += 0.4
            
            # If it's a YouTube lyric video, it's almost certainly bad for background
            if "lyrics" in media_title:
                bad_content_penalty += 1.0

            # Final Score Calculation
            # Consistency is now the DOMINANT factor (4.0x) to ensure relevance.
            # Identity and Reward are secondary motivators.
            score = (reward + emotion + value + identity) + (consistency * 5.0) - (pain * 2.0) - (bad_content_penalty * 3.0)
            return score * attention
            
        except (KeyError, TypeError) as e:
            print(f"[NeuronAI] Error calculating score: {e}")
            return 0.0

    def clear_memory(self):
        """Resets the biological memory in the brain simulator"""
        if self.brain_simulator:
            self.brain_simulator.clear_memory()
