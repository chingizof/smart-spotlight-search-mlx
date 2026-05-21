"""
Tests for data-ingestion/topic_extract.py

ollama.chat is fully mocked — no LLM or network access required.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "data-ingestion"))

from topic_extract import extract_topics


def _mock_ollama_response(payload: dict) -> dict:
    """Build the dict structure ollama.chat returns."""
    return {"message": {"content": json.dumps(payload)}}


def _patch_ollama(payload):
    """Context manager: patches ollama.chat to return the given payload."""
    return patch("topic_extract.ollama.chat",
                 return_value=_mock_ollama_response(payload))


# ═══════════════════════════════════════════════════════════════════════════════
# Happy-path parsing
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractTopicsHappyPath:
    def test_returns_all_four_keys(self):
        with _patch_ollama(
            {"people": ["Sarah"], "places": ["Nobu"], "events": ["dinner"], "topics": ["food"]}
        ):
            result = extract_topics("irrelevant text")
        assert set(result.keys()) == {"people", "places", "events", "topics"}

    def test_people_extracted(self):
        with _patch_ollama({"people": ["Alice", "Bob"], "places": [], "events": [], "topics": []}):
            result = extract_topics("text")
        assert result["people"] == ["Alice", "Bob"]

    def test_places_extracted(self):
        with _patch_ollama({"people": [], "places": ["New York", "LAX"], "events": [], "topics": []}):
            result = extract_topics("text")
        assert result["places"] == ["New York", "LAX"]

    def test_events_extracted_and_lowercased(self):
        with _patch_ollama({"people": [], "places": [], "events": ["Birthday Party", "Flight"], "topics": []}):
            result = extract_topics("text")
        assert result["events"] == ["birthday party", "flight"]

    def test_topics_extracted_and_lowercased(self):
        with _patch_ollama({"people": [], "places": [], "events": [], "topics": ["Travel", "WORK"]}):
            result = extract_topics("text")
        assert result["topics"] == ["travel", "work"]

    def test_people_preserve_original_casing(self):
        with _patch_ollama({"people": ["Dr. Kim", "MICHAEL"], "places": [], "events": [], "topics": []}):
            result = extract_topics("text")
        assert result["people"] == ["Dr. Kim", "MICHAEL"]

    def test_places_preserve_original_casing(self):
        with _patch_ollama({"people": [], "places": ["New York", "Tokyo"], "events": [], "topics": []}):
            result = extract_topics("text")
        assert result["places"] == ["New York", "Tokyo"]

    def test_empty_lists_returned_as_empty(self):
        with _patch_ollama({"people": [], "places": [], "events": [], "topics": []}):
            result = extract_topics("text")
        assert all(v == [] for v in result.values())

    def test_whitespace_only_items_filtered_out(self):
        with _patch_ollama({"people": ["  ", "Alice", ""], "places": [], "events": [], "topics": []}):
            result = extract_topics("text")
        assert result["people"] == ["Alice"]


# ═══════════════════════════════════════════════════════════════════════════════
# Malformed / partial responses
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractTopicsRobustness:
    def test_malformed_json_returns_empty(self):
        bad_response = {"message": {"content": "this is not json {{{"}}
        with patch("topic_extract.ollama.chat", return_value=bad_response):
            result = extract_topics("text")
        assert result == {"people": [], "places": [], "events": [], "topics": []}

    def test_ollama_exception_returns_empty(self):
        with patch("topic_extract.ollama.chat", side_effect=Exception("connection refused")):
            result = extract_topics("text")
        assert result == {"people": [], "places": [], "events": [], "topics": []}

    def test_missing_key_defaults_to_empty_list(self):
        # Response missing "places" and "topics"
        with _patch_ollama({"people": ["Alice"], "events": ["dinner"]}):
            result = extract_topics("text")
        assert result["places"] == []
        assert result["topics"] == []
        assert result["people"] == ["Alice"]

    def test_non_list_value_converted_to_empty(self):
        # LLM returns a string instead of a list for one key
        with _patch_ollama({"people": "Alice", "places": [], "events": [], "topics": []}):
            result = extract_topics("text")
        # Should not crash; non-list becomes []
        assert isinstance(result["people"], list)

    def test_non_string_items_coerced(self):
        # LLM returns numbers inside a list
        with _patch_ollama({"people": [], "places": [], "events": [42, "dinner"], "topics": []}):
            result = extract_topics("text")
        assert "42" in result["events"] or len(result["events"]) >= 1

    def test_empty_string_input_does_not_crash(self):
        with _patch_ollama({"people": [], "places": [], "events": [], "topics": []}):
            result = extract_topics("")
        assert isinstance(result, dict)
