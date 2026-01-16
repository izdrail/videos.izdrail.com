import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from core.nlp.keyword_extractor import KeywordExtractor
from core.nlp.neuron_extractor import NeuronExtractor

class TestNeuronKeywordSearch(unittest.TestCase):

    def setUp(self):
        self.extractor = KeywordExtractor()
        self.neuron_extractor = self.extractor.neuron_extractor

    def test_decision_score_pain_override(self):
        """Test that high pain (Insula) overrides high reward"""
        signals = {
            "attention": 0.8,
            "amygdala": {"salience": 0.9},
            "reward": {"dopamine": 0.9},
            "insula": {"pain": 0.8}, # High pain
            "vmpfc": {"value": 0.5},
            "dlpfc": {"authority": 0.5, "override": False}
        }
        score = self.neuron_extractor._calculate_decision_score(signals)
        self.assertEqual(score, -1.0, "High pain should lead to a -1.0 survival override score")

    def test_decision_score_high_reward(self):
        """Test that high reward/emotion leads to high score"""
        signals = {
            "attention": 0.9,
            "amygdala": {"salience": 0.8},
            "reward": {"dopamine": 0.8},
            "insula": {"pain": 0.1},
            "vmpfc": {"value": 0.7},
            "dlpfc": {"authority": 0.6, "override": True}
        }
        score = self.neuron_extractor._calculate_decision_score(signals)
        self.assertGreater(score, 5.0, "High reward and low pain should yield a high positive score")

    @patch('requests.post')
    def test_extract_keywords_with_neuron_ai(self, mock_post):
        """Test the full flow from KeywordExtractor to NeuronExtractor"""
        # Mock Ollama keyword extraction
        mock_resp_ollama = MagicMock()
        mock_resp_ollama.status_code = 200
        mock_resp_ollama.json.return_value = {"response": "laptop, office, coffee"}
        
        # Mock Neuron evaluation (3 calls, one per candidate)
        mock_resp_neuron = MagicMock()
        mock_resp_neuron.status_code = 200
        # Return different signals for each
        mock_resp_neuron.json.side_effect = [
            {"response": '{"attention": 0.9, "reward": {"dopamine": 0.9}, "insula": {"pain": 0.1}, "vmpfc": {"value": 0.8}, "amygdala": {"salience": 0.8}, "hippocampus": {"consistency": 0.8}, "dlpfc": {"authority": 0.8, "override": false}}'}, # laptop
            {"response": '{"attention": 0.4, "reward": {"dopamine": 0.2}, "insula": {"pain": 0.2}, "vmpfc": {"value": 0.3}, "amygdala": {"salience": 0.2}, "hippocampus": {"consistency": 0.4}, "dlpfc": {"authority": 0.2, "override": false}}'}, # office
            {"response": '{"attention": 0.7, "reward": {"dopamine": 0.5}, "insula": {"pain": 0.8}, "vmpfc": {"value": 0.4}, "amygdala": {"salience": 0.5}, "hippocampus": {"consistency": 0.6}, "dlpfc": {"authority": 0.4, "override": false}}'}  # coffee (high pain)
        ]
        
        mock_post.side_effect = [mock_resp_ollama, mock_resp_neuron, mock_resp_neuron, mock_resp_neuron]
        
        text = "Writing code on a laptop in a busy office while drinking coffee."
        keywords = self.extractor.extract_keywords(text, top_n=2, use_neuron_ai=True)
        
        self.assertIn('laptop', keywords)
        self.assertNotIn('coffee', keywords) # coffee had high pain
        self.assertEqual(len(keywords), 2)
        print(f"\n✅ Neuron AI successfully selected: {keywords}")

if __name__ == "__main__":
    unittest.main()
