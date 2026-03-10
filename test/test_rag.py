"""
Test suite for the RAG system.

Structure:
  - test_document_processor.py  →  chunking logic
  - test_search.py               →  confidence calculation
  - test_responder.py            →  business rules (mocked)
  - test_llm.py                  →  context building and cost
  - test_api.py                  →  HTTP endpoints (FastAPI TestClient)

All in a single file for convenience; split into separate files if the
project grows.

Run with:
    pytest test_rag.py -v
"""

import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Path setup — adjust if your source files live elsewhere
# ---------------------------------------------------------------------------
SRC_DIR = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, SRC_DIR)


# ===========================================================================
# document_processor
# ===========================================================================

class TestSplitIntoChunks:
    """Unit tests for split_into_chunks (pure function, no I/O)."""

    def _split(self, text, size=10, overlap=2):
        from document_processor import split_into_chunks
        return split_into_chunks(text, size=size, overlap=overlap)

    # --- basic behaviour ---------------------------------------------------

    def test_empty_string_returns_empty_list(self):
        assert self._split("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert self._split("   \n\t  ") == []

    def test_text_smaller_than_chunk_size_returns_single_chunk(self):
        words = "one two three four five"
        result = self._split(words, size=10, overlap=2)
        assert len(result) == 1
        assert result[0] == words

    def test_text_exactly_chunk_size_returns_two_chunks_due_to_overlap(self):
        # size=10, overlap=2 → step=8
        # chunk 0: words 0–9, chunk 1: words 8–9 (overlap tail)
        words = " ".join(f"w{i}" for i in range(10))
        result = self._split(words, size=10, overlap=2)
        assert len(result) == 2

    # --- overlap correctness -----------------------------------------------

    def test_overlap_words_appear_in_consecutive_chunks(self):
        # 20 words, size=10, overlap=2  →  step=8
        # chunk 0: words 0-9
        # chunk 1: words 8-17  (words 8 and 9 overlap)
        # chunk 2: words 16-19 (last chunk, shorter)
        words = [f"w{i}" for i in range(20)]
        text = " ".join(words)
        result = self._split(text, size=10, overlap=2)

        chunk0_words = result[0].split()
        chunk1_words = result[1].split()

        # The last `overlap` words of chunk 0 must start chunk 1
        assert chunk0_words[-2:] == chunk1_words[:2]

    def test_overlap_zero_means_no_shared_words(self):
        words = [f"w{i}" for i in range(20)]
        text = " ".join(words)
        result = self._split(text, size=10, overlap=0)

        for i in range(len(result) - 1):
            set_a = set(result[i].split())
            set_b = set(result[i + 1].split())
            assert set_a.isdisjoint(set_b), (
                f"Chunks {i} and {i+1} share words despite overlap=0"
            )

    def test_all_words_are_present_across_chunks(self):
        """Every original word must appear at least once in the output."""
        words = [f"w{i}" for i in range(25)]
        text = " ".join(words)
        result = self._split(text, size=10, overlap=3)

        all_output_words = " ".join(result).split()
        for w in words:
            assert w in all_output_words, f"Word '{w}' missing from chunks"

    # --- edge: overlap >= size would create infinite loop -------------------

    def test_overlap_less_than_size_does_not_hang(self):
        """step = size - overlap must be > 0; just verify it returns."""
        text = " ".join(f"w{i}" for i in range(30))
        result = self._split(text, size=5, overlap=4)   # step = 1
        assert len(result) > 0


class TestReadDocument:
    """Tests for read_document — unsupported extension handling."""

    def test_unknown_extension_returns_empty_string(self, tmp_path):
        from document_processor import read_document
        f = tmp_path / "file.xyz"
        f.write_text("some content")
        assert read_document(str(f)) == ""

    def test_txt_file_is_read_correctly(self, tmp_path):
        from document_processor import read_document
        f = tmp_path / "sample.txt"
        f.write_text("hello world", encoding="utf-8")
        assert read_document(str(f)) == "hello world"


# ===========================================================================
# search  (_calculate_confidence — pure function)
# ===========================================================================

class TestCalculateConfidence:
    """Tests for the distance → (label, score) mapping."""

    def _calc(self, distance):
        from search import _calculate_confidence
        return _calculate_confidence(distance)

    # --- label thresholds (from config: HIGH=0.55, MEDIUM=0.85) -----------

    def test_distance_below_high_threshold_is_high(self):
        label, _ = self._calc(0.30)
        assert label == "high"

    def test_distance_at_high_threshold_is_medium(self):
        # distance == HIGH_CONFIDENCE_THRESHOLD  → NOT < threshold → medium
        label, _ = self._calc(0.55)
        assert label == "medium"

    def test_distance_between_thresholds_is_medium(self):
        label, _ = self._calc(0.70)
        assert label == "medium"

    def test_distance_at_medium_threshold_is_low(self):
        label, _ = self._calc(0.85)
        assert label == "low"

    def test_distance_above_medium_threshold_is_low(self):
        label, _ = self._calc(1.50)
        assert label == "low"

    # --- score (percentage) ------------------------------------------------

    def test_score_is_between_0_and_1(self):
        for distance in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
            _, score = self._calc(distance)
            assert 0.0 <= score <= 1.0, f"score out of range for distance={distance}"

    def test_score_never_negative_for_large_distance(self):
        _, score = self._calc(999.0)
        assert score == 0.0

    def test_zero_distance_gives_score_1(self):
        _, score = self._calc(0.0)
        assert score == 1.0

    def test_score_decreases_as_distance_increases(self):
        _, s1 = self._calc(0.2)
        _, s2 = self._calc(0.6)
        _, s3 = self._calc(1.2)
        assert s1 > s2 > s3


# ===========================================================================
# llm  (pure functions — no API calls)
# ===========================================================================

class TestBuildContext:
    """Tests for build_context."""

    def _build(self, chunks):
        from llm import build_context
        return build_context(chunks)

    def test_single_chunk_contains_source_and_excerpt(self):
        chunks = [{"file": "doc_a.pdf", "excerpt": "some text here"}]
        ctx = self._build(chunks)
        assert "doc_a.pdf" in ctx
        assert "some text here" in ctx

    def test_multiple_chunks_are_numbered_sequentially(self):
        chunks = [
            {"file": "a.pdf", "excerpt": "first"},
            {"file": "b.pdf", "excerpt": "second"},
            {"file": "c.pdf", "excerpt": "third"},
        ]
        ctx = self._build(chunks)
        assert "EXCERPT 1" in ctx
        assert "EXCERPT 2" in ctx
        assert "EXCERPT 3" in ctx

    def test_chunks_separated_by_blank_line(self):
        chunks = [
            {"file": "a.pdf", "excerpt": "alpha"},
            {"file": "b.pdf", "excerpt": "beta"},
        ]
        ctx = self._build(chunks)
        assert "\n\n" in ctx


class TestCalculateCost:
    """Tests for calculate_cost."""

    def _cost(self, inp, out):
        from llm import calculate_cost
        return calculate_cost(inp, out)

    def test_zero_tokens_returns_zero_costs(self):
        c = self._cost(0, 0)
        assert c["total_cost_eur"] == 0.0
        assert c["input_cost_eur"] == 0.0
        assert c["output_cost_eur"] == 0.0

    def test_token_counts_are_preserved(self):
        c = self._cost(100, 200)
        assert c["tokens_input"] == 100
        assert c["tokens_output"] == 200

    def test_total_equals_input_plus_output(self):
        c = self._cost(500, 300)
        assert abs(c["total_cost_eur"] - (c["input_cost_eur"] + c["output_cost_eur"])) < 1e-9

    def test_output_more_expensive_per_token_than_input(self):
        # From config: INPUT=0.05/1M, OUTPUT=0.08/1M
        c_in = self._cost(1_000_000, 0)
        c_out = self._cost(0, 1_000_000)
        assert c_out["total_cost_eur"] > c_in["total_cost_eur"]

    def test_cost_values_are_rounded_to_6_decimal_places(self):
        c = self._cost(123456, 789012)
        for key in ("input_cost_eur", "output_cost_eur", "total_cost_eur"):
            val = c[key]
            assert round(val, 6) == val


# ===========================================================================
# responder  (business rules — mocked search + llm)
# ===========================================================================

# Helpers ------------------------------------------------------------------

def _make_chunk(file="doc.pdf", excerpt="some text", distance=0.30,
                confidence="high", score=0.85):
    return {
        "file": file,
        "excerpt": excerpt,
        "distance": distance,
        "confidence": confidence,
        "score": score,
        "chunk_index": 0,
    }


def _zero_cost():
    from llm import calculate_cost
    return calculate_cost(0, 0)


def _some_cost():
    from llm import calculate_cost
    return calculate_cost(100, 50)


# --------------------------------------------------------------------------

class TestResponderRefusals:
    """Business rules: when should the responder refuse?"""

    def test_no_chunks_returns_refused(self):
        from responder import respond
        with patch("responder.search", return_value=[]):
            result = respond("what is the return policy?")
        assert result["refused"] is True
        assert result["confidence"] == "insufficient"

    def test_no_chunks_does_not_call_llm(self):
        from responder import respond
        with patch("responder.search", return_value=[]):
            with patch("responder.complete") as mock_llm:
                respond("anything")
        mock_llm.assert_not_called()

    def test_score_below_threshold_refuses_without_calling_llm(self):
        from responder import respond
        low_score_chunk = _make_chunk(score=0.10, distance=1.80, confidence="low")

        with patch("responder.search", return_value=[low_score_chunk]):
            with patch("responder.complete") as mock_llm:
                result = respond("some question")

        mock_llm.assert_not_called()
        assert result["refused"] is True

    def test_llm_refusal_phrase_downgrades_confidence(self):
        from responder import respond
        chunk = _make_chunk(score=0.90)
        refusal_text = "I could not find this information in the documents."

        with patch("responder.search", return_value=[chunk]):
            with patch("responder.complete", return_value=(refusal_text, _some_cost())):
                result = respond("some question")

        assert result["refused"] is True
        assert result["confidence"] == "insufficient"

    def test_llm_refusal_detection_is_case_insensitive(self):
        from responder import respond
        chunk = _make_chunk(score=0.90)
        mixed_case = "i Could Not Find This Information in the documents."

        with patch("responder.search", return_value=[chunk]):
            with patch("responder.complete", return_value=(mixed_case, _some_cost())):
                result = respond("question")

        assert result["refused"] is True


class TestResponderSuccess:
    """Business rules: normal (non-refused) response path."""

    def _successful_respond(self, confidence="high", score=0.90):
        from responder import respond
        chunk = _make_chunk(confidence=confidence, score=score)
        answer_text = "The answer is 42."

        with patch("responder.search", return_value=[chunk]):
            with patch("responder.complete", return_value=(answer_text, _some_cost())):
                return respond("what is the answer?")

    def test_successful_response_is_not_refused(self):
        assert self._successful_respond()["refused"] is False

    def test_successful_response_contains_answer(self):
        result = self._successful_respond()
        assert result["answer"] == "The answer is 42."

    def test_confidence_comes_from_first_chunk(self):
        result = self._successful_respond(confidence="medium")
        assert result["confidence"] == "medium"

    def test_sources_list_is_sorted_and_deduplicated(self):
        from responder import respond
        chunks = [
            _make_chunk(file="z.pdf", score=0.90),
            _make_chunk(file="a.pdf", score=0.88),
            _make_chunk(file="z.pdf", score=0.85),  # duplicate
        ]
        with patch("responder.search", return_value=chunks):
            with patch("responder.complete", return_value=("ok", _some_cost())):
                result = respond("question")

        assert result["sources"] == ["a.pdf", "z.pdf"]

    def test_average_score_is_correct(self):
        from responder import respond
        chunks = [
            _make_chunk(score=0.80),
            _make_chunk(score=0.60),
        ]
        with patch("responder.search", return_value=chunks):
            with patch("responder.complete", return_value=("answer", _some_cost())):
                result = respond("question")

        assert result["average_score"] == round((0.80 + 0.60) / 2, 4)

    def test_cost_is_passed_through(self):
        from responder import respond
        chunk = _make_chunk(score=0.90)
        cost = _some_cost()

        with patch("responder.search", return_value=[chunk]):
            with patch("responder.complete", return_value=("ans", cost)):
                result = respond("question")

        assert result["cost"]["tokens_input"] == cost["tokens_input"]
        assert result["cost"]["tokens_output"] == cost["tokens_output"]


# ===========================================================================
# api  (HTTP layer — FastAPI TestClient, fully mocked internals)
# ===========================================================================

class TestAPIStatus:

    def _client(self):
        from fastapi.testclient import TestClient
        import api
        return TestClient(api.app)

    def test_status_returns_200(self):
        with patch("api.get_collection") as mock_col:
            mock_col.return_value.count.return_value = 42
            resp = self._client().get("/status")
        assert resp.status_code == 200

    def test_status_contains_expected_fields(self):
        with patch("api.get_collection") as mock_col:
            mock_col.return_value.count.return_value = 7
            resp = self._client().get("/status")
        data = resp.json()
        assert "status" in data
        assert "chunks_in_db" in data
        assert "embedding_model" in data
        assert "llm_model" in data

    def test_status_chunk_count_matches_collection(self):
        with patch("api.get_collection") as mock_col:
            mock_col.return_value.count.return_value = 99
            resp = self._client().get("/status")
        assert resp.json()["chunks_in_db"] == 99


class TestAPIAsk:

    def _client(self):
        from fastapi.testclient import TestClient
        import api
        return TestClient(api.app)

    def _mock_respond(self, answer="The answer.", refused=False, confidence="high"):
        from llm import calculate_cost
        return {
            "answer": answer,
            "sources": ["doc.pdf"],
            "chunks": [{"file": "doc.pdf", "excerpt": "text", "score": 0.9, "confidence": confidence}],
            "confidence": confidence,
            "average_score": 0.9,
            "cost": calculate_cost(100, 50),
            "refused": refused,
        }

    # --- validation --------------------------------------------------------

    def test_question_shorter_than_5_chars_is_refused(self):
        resp = self._client().post("/ask", json={"question": "hi"})
        assert resp.status_code == 200
        assert resp.json()["refused"] is True

    def test_question_of_exactly_4_chars_is_refused(self):
        resp = self._client().post("/ask", json={"question": "abcd"})
        assert resp.json()["refused"] is True

    def test_question_of_5_chars_is_processed(self):
        with patch("api.respond", return_value=self._mock_respond()):
            resp = self._client().post("/ask", json={"question": "abcde"})
        assert resp.json()["refused"] is False

    # --- normal flow -------------------------------------------------------

    def test_successful_ask_returns_200(self):
        with patch("api.respond", return_value=self._mock_respond()):
            resp = self._client().post("/ask", json={"question": "What is the refund policy?"})
        assert resp.status_code == 200

    def test_answer_is_forwarded_correctly(self):
        with patch("api.respond", return_value=self._mock_respond(answer="42 days.")):
            resp = self._client().post("/ask", json={"question": "How long is the warranty?"})
        assert resp.json()["answer"] == "42 days."

    def test_refused_flag_is_forwarded(self):
        with patch("api.respond", return_value=self._mock_respond(refused=True, confidence="insufficient")):
            resp = self._client().post("/ask", json={"question": "Unknown topic here"})
        assert resp.json()["refused"] is True

    # --- error handling ----------------------------------------------------

    def test_internal_exception_returns_refused_response_not_500(self):
        with patch("api.respond", side_effect=RuntimeError("boom")):
            resp = self._client().post("/ask", json={"question": "This will fail internally"})
        assert resp.status_code == 200          # must NOT be 500
        data = resp.json()
        assert data["refused"] is True
        assert "boom" in data["answer"]

    # --- response schema ---------------------------------------------------

    def test_response_contains_all_expected_fields(self):
        with patch("api.respond", return_value=self._mock_respond()):
            resp = self._client().post("/ask", json={"question": "Valid question here?"})
        data = resp.json()
        for field in ("answer", "sources", "chunks", "confidence", "average_score", "cost", "refused"):
            assert field in data, f"Missing field: {field}"

    def test_cost_contains_all_expected_fields(self):
        with patch("api.respond", return_value=self._mock_respond()):
            resp = self._client().post("/ask", json={"question": "Valid question here?"})
        cost = resp.json()["cost"]
        for field in ("tokens_input", "tokens_output", "total_cost_eur"):
            assert field in cost, f"Missing cost field: {field}"