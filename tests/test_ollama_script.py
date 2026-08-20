import sys
import os
from unittest.mock import MagicMock, patch
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.nlp.keyword_extractor import KeywordExtractor

def test_generate_script():
    print("Testing generate_script_from_text...")
    
    # Mock response from Ollama
    mock_response = {
        "response": "Here is the clean script without pauses."
    }
    
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = mock_response
        
        extractor = KeywordExtractor()
        # Set dummy url to avoid env var issues if any
        extractor.api_url = "http://localhost:11434"
        
        input_text = "Some raw text with [pause] and instructions."
        result = extractor.generate_script_from_text(input_text)
        
        print(f"Input: {input_text}")
        print(f"Result: {result}")
        
        # Verify the call
        args, kwargs = mock_post.call_args
        payload = kwargs['json']
        
        assert payload['model'] == "gemma4:e2b"
        assert "Generate a tts readys script no [pause]" in payload['prompt']
        assert result == "Here is the clean script without pauses."
        print("✅ Test passed!")

if __name__ == "__main__":
    test_generate_script()
