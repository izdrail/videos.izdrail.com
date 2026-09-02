"""
Prompt Generator for SD-Turbo Image Generation.
Converts context sentences, keywords, and entities into high quality prompts.
"""
import re
from typing import List, Optional


class PromptGenerator:
    """Generates SD-Turbo image prompts from sentence context and metadata."""

    TEMPLATES = {
        "business": "professional corporate environment, modern office, meeting, {concept}",
        "technology": "futuristic technology, digital innovation, {concept}",
        "nature": "serene natural landscape, outdoor, {concept}",
        "people": "diverse group of people, authentic emotions, {concept}",
        "default": "cinematic scene, {concept}",
    }

    STYLE_MODIFIERS = [
        "documentary photography",
        "dramatic natural lighting",
        "vertical social media composition",
        "professional grade",
        "vivid colors",
        "sharp focus",
    ]

    def __init__(self, config=None):
        self.config = config

    def generate(
        self,
        sentence: Optional[str] = None,
        keyword: Optional[str] = None,
        entities: Optional[List[str]] = None,
    ) -> str:
        """Generate a prompt suitable for SD-Turbo text-to-image pipeline."""
        concepts = self._extract_concepts(sentence, keyword, entities)

        base_concept = ", ".join(concepts) if concepts else "a captivating visual scene"

        base_prompt = f"Cinematic editorial photograph of {base_concept}"
        modifiers = ", ".join(self.STYLE_MODIFIERS)

        return f"{base_prompt}, {modifiers}"

    def _extract_concepts(
        self,
        sentence: Optional[str],
        keyword: Optional[str],
        entities: Optional[List[str]],
    ) -> List[str]:
        concepts = []

        if keyword and keyword.strip():
            concepts.append(keyword.strip())

        if entities:
            for ent in entities:
                if ent and ent.strip() and ent.strip() not in concepts:
                    concepts.append(ent.strip())

        if sentence and not concepts:
            cleaned = re.sub(r"[^\w\s]", "", sentence).strip()
            words = [w for w in cleaned.split() if len(w) > 3]
            if words:
                concepts.append(" ".join(words[:4]))

        return concepts
