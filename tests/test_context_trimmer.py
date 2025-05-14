import sys
print("PYTEST sys.path:", sys.path)
import pytest
from forest_app.integrations.context_trimmer import ContextTrimmer, TrimmerConfig

def test_token_count_basic():
    trimmer = ContextTrimmer()
    text = "Hello world!"
    tokens = trimmer.count_tokens(text)
    assert isinstance(tokens, int)
    assert tokens > 0

def test_trim_preserves_structure():
    config = TrimmerConfig(max_tokens=10, buffer_tokens=0, preserve_recent_ratio=1.0, preserve_first_n_chars=0)
    trimmer = ContextTrimmer(config)
    # Simulate sections with repetitive content
    content = "Section1\n" + ("a " * 50) + "\nSection2\n" + ("b " * 50)
    trimmed, _ = trimmer.trim_content(content)
    # Should not exceed max_tokens
    assert trimmer.count_tokens(trimmed) <= config.max_tokens
    # Should preserve at least one section header, given the tight token limit
    assert "Section1" in trimmed or "Section2" in trimmed
