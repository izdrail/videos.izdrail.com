"""
Keyword Extraction
Uses AI and NLP to extract visual keywords from text
"""
import os
import re
import requests
from typing import List, Optional
from collections import Counter

class OllamaKeywordExtractor:
    """Uses Ollama API to extract keywords from text"""
    
    def __init__(self, model: str = "mistral:7b"):
        self.model = model
        self.url = os.getenv("OLLAMA_API_URL", "https://ai.izdrail.com/api/generate")
        self.cache = {}
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        cache_key = f"{text[:100]}_{top_n}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        prompt = (
            f"Extract up to {top_n} relevant, concrete, visual keywords from this sentence. "
            "Return only a comma-separated list of lowercase words or short phrases (max 3 words each). "
            "Avoid abstract concepts, stop words, or brand names.\n"
            f"Sentence: \"{text}\"\nKeywords:"
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 64}
        }
        try:
            response = requests.post(self.url, json=payload, timeout=20)
            if response.status_code == 200:
                raw = response.json().get("response", "").strip()
                # Clean and parse
                keywords = [kw.strip().lower() for kw in raw.split(",") if kw.strip()]
                # Further cleaning: remove non-alphanumeric (except spaces)
                keywords = [re.sub(r'[^a-zA-Z0-9\s]', '', kw) for kw in keywords]
                result = [kw for kw in keywords if kw][:top_n]
                self.cache[cache_key] = result
                return result
        except Exception as e:
            print(f"[Ollama] Error extracting keywords: {e}")
        return []

class KeywordExtractor:
    """Orchestrates keyword extraction using Ollama and Spacy fallback"""
    
    def __init__(self, ollama_model: str = "mistral:7b"):
        self.ollama_extractor = OllamaKeywordExtractor(model=ollama_model)
        self.relevant_pos = {'NOUN', 'PROPN', 'ADJ'}
        self.exclude_words = {
            'thing', 'things', 'something', 'someone', 'way', 'time', 'day',
            'year', 'week', 'month', 'people', 'person', 'place', 'lot',
            'intro', 'outro', 'welcome', 'thanks', 'watching', 'subscribe'
        }
        self.used_keywords = set()
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """Main entry point for keyword extraction"""
        if not text.strip():
            return []
            
        # Try Ollama first
        ollama_keywords = self.ollama_extractor.extract_keywords(text, top_n)
        if ollama_keywords:
            return ollama_keywords
        
        # Fallback to Spacy API
        return self._extract_spacy_fallback(text, top_n)
    
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
        """Reset used keywords tracker"""
        self.used_keywords.clear()
        
    def sanitize_keyword(self, keyword: str) -> str:
        """Clean a keyword for API search"""
        if not keyword: return ""
        kw = re.sub(r'[\*\-•\n]+', '', keyword)
        return re.sub(r'\s+', ' ', kw).strip().lower()
