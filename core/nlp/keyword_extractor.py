"""
Keyword Extraction
Uses AI and NLP to extract visual keywords from text
"""
import os
import re
import requests
from typing import List, Optional
from collections import Counter
import spacy
from collections import Counter
from .neuron_extractor import NeuronExtractor

class OllamaKeywordExtractor:
    """Uses Ollama API to extract keywords from text"""
    
    def __init__(self, model: str = "mistral:7b", url: Optional[str] = None):
        self.model = model
        self.url = url or os.getenv("OLLAMA_API_URL", "https://ai.izdrail.com/api/generate")
        self.cache = {}
    
    def extract_keywords(self, text: str, top_n: int = 5, language: str = 'en') -> List[str]:
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
    f"Text: \"{text}\"\n"
    f"Keywords:"
)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 64}
        }
        try:
            response = requests.post(self.url, json=payload, timeout=180)
            if response.status_code == 200:
                raw = response.json().get("response", "").strip()
                print(f"[Ollama] Raw output: {raw}")
                # Clean and parse
                raw_keywords = [kw.strip().lower() for kw in raw.split(",") if kw.strip()]
                # Further cleaning: remove non-alphanumeric (except spaces)
                keywords = [re.sub(r'[^a-zA-Z0-9\s]', '', kw) for kw in raw_keywords]
                # Allow 1-2 word keywords
                filtered_keywords = []
                for kw in keywords:
                    words = kw.split()
                    if len(words) == 1 and words[0]:
                        filtered_keywords.append(words[0])
                    elif len(words) == 2:
                        # Keep 2-word phrases
                        filtered_keywords.append(f"{words[0]} {words[1]}")
                    elif len(words) > 2:
                        # Take first 2 words if longer
                        filtered_keywords.append(f"{words[0]} {words[1]}")
                
                # Filter out empty and limit to top_n
                result = filtered_keywords[:top_n]
                self.cache[cache_key] = result
                return result
        except Exception as e:
            print(f"[Ollama] Error extracting keywords: {e}")
        return []

    def generate_social_media_descriptions(self, text: str, keywords: List[str], language: str = 'en') -> str:
        """
        Generates social media descriptions (YouTube Title, Description, Hashtags)
        using Ollama based on the video script and keywords.
        """
        try:
            prompt = f"""
            You are a professional social media manager. 
            Based on the following video script and extracted keywords, create:
            1. A catchy YouTube Video Title (max 60 chars)
            2. A compelling Video Description (max 200 chars)
            3. A list of 10 relevant hashtags
            4. A short TikTok/Reels caption (max 100 chars)

            Script: "{text[:1000]}..."
            Keywords: {', '.join(keywords)}
            Language: {language}

            Output format:
            Title: [Title]
            Description: [Description]
            Hashtags: #tag1 #tag2 ...
            TikTok: [Caption]
            """

            response = requests.post(
                self.url, # Changed from f"{self.base_url}/api/generate" to self.url
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            return "Failed to generate descriptions."
            
        except Exception as e:
            print(f"Error generating descriptions: {e}")
            return f"Error: {str(e)}"

    def generate_script_from_text(self, text: str) -> str:
        """
        Generates a clean TTS ready script from raw text using Ollama.
        Removes [pause] and other non-spoken instructions.
        """
        try:
            prompt = f"Generate a tts readys script no [pause] or anything like that from the following text: {text}"

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7
                }
            }

            response = requests.post(
                self.url, # Changed from f"{self.base_url}/api/generate" to self.url
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            return text # Return original text on failure
            
        except Exception as e:
            print(f"Error generating script: {e}")
            return text

    @staticmethod
    def fetch_models_static(base_url: str) -> List[str]:
        """Fetches available models from the current instance's API URL."""
         # Re-implementing fetch_models_static logic since I seemingly overwrote/messed up get_available_models and the static method below it might be affected?
         # Wait, looking at lines 92-100 in the view, it seems I cut into `get_available_models` docstring or similar?
         # "Fetches available models from the current instance's API URL." is the docstring for `fetch_models_static` usually.
         # Let me check what was there before or just fix it to be correct code.
         # lines 92+ look like the start of `fetch_models_static` but the def line is missing.
         
        if "/generate" in base_url:
            url = base_url.replace("/generate", "/tags")
        else:
            # Try to append /tags if not present
            from urllib.parse import urlparse
            p = urlparse(base_url)
            # If path ends with /api/generate or similar, fix it
            url = f"{p.scheme}://{p.netloc}/api/tags"

        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = [m['name'] for m in data.get('models', [])]
                return models
        except:
            pass
        return []

    def get_available_models(self) -> List[str]:
        models = self.fetch_models_static(self.url)
        # Ensure current model and default model are in the list
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
        
        # Initialize local spaCy for instant extraction
        try:
            self.nlp = spacy.load("en_core_web_md")
        except:
            # Fallback to sm if md is not available
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                self.nlp = None
                print("[NLP] Warning: Local spaCy model not found. Using fallback methods.")

        self.relevant_pos = {'NOUN', 'PROPN', 'ADJ'}
        self.exclude_words = {
            'thing', 'things', 'something', 'someone', 'way', 'time', 'day',
            'year', 'week', 'month', 'people', 'person', 'place', 'lot',
            'intro', 'outro', 'welcome', 'thanks', 'watching', 'subscribe'
        }
        self.used_keywords = set()
    
    def extract_keywords(self, text: str, top_n: int = 5, language: str = 'en', use_neuron_ai: bool = True, use_snn: bool = False) -> List[str]:
        """Main entry point for keyword extraction. Prioritizes local spaCy for speed."""
        if not text.strip():
            return []
            
        # 1. Try local spaCy extraction (INSTANT)
        candidates = self._extract_spacy_local(text, top_n * 2)
        
        # 2. If spaCy fails or returns too few, fallback to Ollama (SLOW)
        if len(candidates) < 2 and language == 'en':
            ollama_keywords = self.ollama_extractor.extract_keywords(text, min(4, top_n * 2), language)
            candidates.extend([k for k in ollama_keywords if k not in candidates])

        if candidates:
            if use_neuron_ai:
                # Use local vector evaluation for max speed
                neuron_results = self.neuron_extractor.evaluate_keywords(text, candidates, language, use_snn=use_snn)
                if neuron_results:
                    return [res['keyword'] for res in neuron_results[:top_n]]
            
            # Rank and return top_n
            ranked = self.rank_keywords(candidates)
            return ranked[:top_n]
        
        return []

    def _extract_spacy_local(self, text: str, top_n: int = 5) -> List[str]:
        """Extract keywords using local Spacy model"""
        if not self.nlp:
            return []
            
        doc = self.nlp(text.lower())
        candidates = []
        
        # 1. Prioritize Entities (Locations, Orgs, Products)
        for ent in doc.ents:
            if ent.label_ in {"GPE", "LOC", "ORG", "PRODUCT", "EVENT", "PERSON"}:
                candidates.append(ent.text)
        
        # 2. Extract Noun Phrases (2-3 words) - EXTREMELY HIGH VALUE for search
        # We skip chunks that are just stop words or too long.
        for chunk in doc.noun_chunks:
            # Clean the chunk text (remove front/back articles/stop words)
            clean_chunk = " ".join([t.text for t in chunk if not t.is_stop and t.pos_ in {"NOUN", "PROPN", "ADJ"}])
            if clean_chunk and len(clean_chunk.split()) >= 1:
                # Prioritize multi-word phrases by adding them first
                candidates.append(clean_chunk)
                
        # 3. Add high-value individual tokens
        for token in doc:
            if (token.pos_ in self.relevant_pos and 
                not token.is_stop and 
                len(token.text) > 2 and 
                token.text.isalpha() and 
                token.text not in self.exclude_words):
                candidates.append(token.text)
        
        if not candidates:
            return []
            
        # Count and get most common, preserving order for entities
        counts = Counter(candidates)
        return [word for word, count in counts.most_common(top_n)]

    
    def _extract_spacy_fallback(self, text: str, top_n: int = 5) -> List[str]:
        """Fallback to external Spacy API if Ollama fails"""
        spacy_url = os.getenv("SPACY_API_URL", "https://spacy.izdrail.com")
        try:
            pos_resp = requests.post(f"{spacy_url}/pos", json={"text": text.lower()}, timeout=10)
            candidates = []
            if pos_resp.status_code == 200:
                tokens = pos_resp.json()
                for token in tokens:
                    pos = token.get('pos')
                    word = token.get('text')
                    is_stop = token.get('is_stop', False)
                    if (pos in self.relevant_pos and 
                        not is_stop and 
                        len(word) > 2 and 
                        word.isalpha() and 
                        word not in self.exclude_words):
                        candidates.append(word)
            
            if not candidates:
                return []
                
            freq = Counter(candidates)
            return [word for word, count in freq.most_common(top_n)]
        except Exception as e:
            print(f"[NLP] Spacy fallback error: {e}")
            return []

    def get_best_unique_keyword(self, text: str, language: Optional[str] = None) -> Optional[str]:
        """Get a keyword that hasn't been used yet in this generation session"""
        keywords = self.extract_keywords(text, top_n=10)
        for kw in keywords:
            if kw not in self.used_keywords:
                self.used_keywords.add(kw)
                return kw
        return keywords[0] if keywords else None

    def clear_used(self):
        """Reset used keywords tracker and biological memory"""
        self.used_keywords.clear()
        self.neuron_extractor.clear_memory()
    
    def rank_keywords(self, keywords: List[str]) -> List[str]:
        """
        Rank keywords by quality/searchability.
        Prioritizes:
        - Common stock footage categories
        - Visual/concrete terms
        - Unused keywords
        """
        # Common stock footage categories (high priority)
        stock_categories = {
            'nature', 'forest', 'ocean', 'mountain', 'sky', 'sunset', 'sunrise',
            'city', 'cityscape', 'building', 'office', 'street', 'traffic',
            'people', 'business', 'meeting', 'technology', 'computer', 'phone',
            'food', 'cooking', 'restaurant', 'travel', 'beach', 'landscape',
            'water', 'fire', 'clouds', 'rain', 'snow', 'night', 'day',
            'abstract', 'motion', 'light', 'color', 'texture', 'pattern',
            'hands', 'work', 'team', 'collaboration', 'innovation', 'growth'
        }
        
        def score_keyword(kw: str) -> float:
            score = 0.0
            kw_lower = kw.lower()
            
            # Bonus for unused keywords
            if kw not in self.used_keywords:
                score += 10.0
            
            # Bonus for stock footage categories
            for category in stock_categories:
                if category in kw_lower:
                    score += 5.0
                    break
            
            # Bonus for 2-word phrases (more specific)
            if len(kw.split()) == 2:
                score += 3.0
            
            # Penalty for very generic terms
            generic_terms = {'thing', 'stuff', 'item', 'object', 'concept', 'idea'}
            if kw_lower in generic_terms:
                score -= 5.0
            
            # Bonus for visual action words
            visual_actions = {'moving', 'flowing', 'growing', 'building', 'working', 'walking', 'running', 'flying'}
            if any(action in kw_lower for action in visual_actions):
                score += 2.0
            
            return score
        
        # Sort by score (descending)
        ranked = sorted(keywords, key=score_keyword, reverse=True)
        return ranked
    
    def generate_fallback_keywords(self, keyword: str) -> List[str]:
        """
        Generate fallback keywords for when primary search fails.
        Returns broader/related terms.
        """
        fallbacks = []
        kw_lower = keyword.lower()
        
        # Category mappings for common terms
        category_map = {
            # Nature
            'tree': ['forest', 'nature', 'woods'],
            'flower': ['garden', 'nature', 'plants'],
            'garden': ['nature', 'plants', 'outdoor'],
            'ocean': ['water', 'sea', 'waves'],
            'mountain': ['landscape', 'nature', 'outdoor'],
            'river': ['water', 'nature', 'stream'],
            
            # Urban
            'building': ['city', 'architecture', 'urban'],
            'office': ['business', 'workplace', 'indoor'],
            'street': ['city', 'urban', 'traffic'],
            'car': ['traffic', 'transportation', 'vehicle'],
            
            # Technology
            'computer': ['technology', 'office', 'digital'],
            'phone': ['technology', 'communication', 'mobile'],
            'code': ['technology', 'programming', 'digital'],
            'data': ['technology', 'digital', 'abstract'],
            
            # People/Activities
            'meeting': ['business', 'people', 'office'],
            'team': ['business', 'people', 'collaboration'],
            'work': ['business', 'office', 'people'],
            'cooking': ['food', 'kitchen', 'chef'],
        }
        
        # Check if keyword or its parts are in category map
        for key, values in category_map.items():
            if key in kw_lower:
                fallbacks.extend(values)
                break
        
        # If multi-word, try first word only
        words = keyword.split()
        if len(words) > 1:
            fallbacks.append(words[0])
        
        # Generic safe fallbacks
        generic_fallbacks = ['abstract', 'motion', 'light', 'texture', 'landscape', 'cityscape']
        
        # Return unique fallbacks
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
        """Clean a keyword for API search"""
        if not keyword: return ""
        kw = re.sub(r'[\*\-•\n]+', '', keyword)
        return re.sub(r'\s+', ' ', kw).strip().lower()

    def get_available_models(self) -> List[str]:
        """Proxy to Ollama extractor"""
        return self.ollama_extractor.get_available_models()

    @property
    def api_url(self) -> str:
        return self.ollama_extractor.url

    @api_url.setter
    def api_url(self, value: str):
        if value and value != self.ollama_extractor.url:
            print(f"[Ollama] Updating API URL to: {value}")
            self.ollama_extractor.url = value
            self.ollama_extractor.cache.clear() # Clear cache on URL change

    @property
    def model(self) -> str:
        return self.ollama_extractor.model

    @model.setter
    def model(self, value: str):
        self.ollama_extractor.model = value

    def generate_social_media_descriptions(self, text: str, keywords: List[str], language: str = 'en') -> str:
        return self.ollama_extractor.generate_social_media_descriptions(text, keywords, language)

    def generate_script_from_text(self, text: str) -> str:
        return self.ollama_extractor.generate_script_from_text(text)

