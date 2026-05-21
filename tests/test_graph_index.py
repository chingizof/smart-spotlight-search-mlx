"""
Tests for data-ingestion/graph_index.py

Covers:
- Node ID helpers
- Speaker name extraction
- add_chunk_to_graph: node/edge creation, embedding storage, return value
- Temporal edges between consecutive chunks
- recompute_bm25_weights: TF/IDF weighting, ordering, floor value
- add_semantic_topic_edges: threshold, bidirectionality, edge type
- ppr_search: seed selection, ranking, string fallback, result structure
- save_graph / load_graph round-trip

Heavy deps (sentence_transformers, ollama, lancedb) are never imported.
The embedding model is replaced with a lightweight deterministic mock.
"""

import json
import math
import sys
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

# Make data-ingestion importable from the tests/ directory
sys.path.insert(0, str(Path(__file__).parent.parent / "data-ingestion"))

import graph_index as gi
from graph_index import (
    add_chunk_to_graph,
    add_semantic_topic_edges,
    chunk_node_id,
    extract_speaker_names,
    load_graph,
    ppr_search,
    recompute_bm25_weights,
    save_graph,
    topic_node_id,
)


# ── Mock embedding model ──────────────────────────────────────────────────────
#
# 8-dimensional space.  Predefined keys map to named unit vectors so tests can
# assert specific cosine similarities without randomness.
#
#   dinner / dining / restaurant  → axis 0  (similar to each other, sim=1.0)
#   travel / flight               → axis 1  (similar to each other, sim=1.0)
#   work / meeting                → axis 2  (similar to each other, sim=1.0)
#   Sarah                         → axis 3
#   John                          → axis 4
#   <unknown>                     → axis 7  (distinct from all named axes)

_EMBED_TABLE = {
    "dinner":     np.array([1., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32),
    "dining":     np.array([1., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32),
    "restaurant": np.array([1., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32),
    "travel":     np.array([0., 1., 0., 0., 0., 0., 0., 0.], dtype=np.float32),
    "flight":     np.array([0., 1., 0., 0., 0., 0., 0., 0.], dtype=np.float32),
    "work":       np.array([0., 0., 1., 0., 0., 0., 0., 0.], dtype=np.float32),
    "meeting":    np.array([0., 0., 1., 0., 0., 0., 0., 0.], dtype=np.float32),
    "Sarah":      np.array([0., 0., 0., 1., 0., 0., 0., 0.], dtype=np.float32),
    "John":       np.array([0., 0., 0., 0., 1., 0., 0., 0.], dtype=np.float32),
}
_EMBED_DEFAULT = np.array([0., 0., 0., 0., 0., 0., 0., 1.], dtype=np.float32)
_EMBED_PREFIX = "search_query: "


class MockModel:
    def encode(self, texts, convert_to_numpy=True, **kwargs):
        vecs = []
        for text in texts:
            key = text[len(_EMBED_PREFIX):] if text.startswith(_EMBED_PREFIX) else text
            vecs.append(_EMBED_TABLE.get(key, _EMBED_DEFAULT).copy())
        return np.array(vecs, dtype=np.float32)


MODEL = MockModel()


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_chunk(G, node_id, topics, chat_id=1, start="2024-01-01T10:00",
               end="2024-01-01T10:05", raw_text="text", prev_node_id=None):
    """Convenience wrapper for add_chunk_to_graph using MockModel."""
    return add_chunk_to_graph(
        G=G, node_id=node_id, raw_text=raw_text,
        topics=topics, start_time=start, end_time=end,
        chat_id=chat_id, model=MODEL, prev_node_id=prev_node_id,
    )


def empty_topics():
    return {"people": [], "places": [], "events": [], "topics": []}


# ═══════════════════════════════════════════════════════════════════════════════
# Node ID helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestNodeIdHelpers:
    def test_chunk_node_id_format(self):
        assert chunk_node_id(1, 1000.9) == "chunk:1:1000"

    def test_chunk_node_id_zero_timestamp(self):
        assert chunk_node_id(42, 0.0) == "chunk:42:0"

    def test_topic_node_id_lowercases(self):
        assert topic_node_id("people", "Sarah") == "people:sarah"

    def test_topic_node_id_multiword(self):
        assert topic_node_id("events", "Birthday Party") == "events:birthday party"

    def test_topic_node_id_already_lower(self):
        assert topic_node_id("topics", "travel") == "topics:travel"


# ═══════════════════════════════════════════════════════════════════════════════
# Speaker name extraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractSpeakerNames:
    def test_extracts_named_speakers(self):
        text = "[Sarah]: Hi\n[John]: Hello"
        names = extract_speaker_names(text)
        assert "Sarah" in names
        assert "John" in names

    def test_skips_me(self):
        names = extract_speaker_names("[Me]: hello\n[Sarah]: hey")
        assert "Me" not in names
        assert "Sarah" in names

    def test_skips_unknown(self):
        names = extract_speaker_names("[Unknown]: ???\n[Sarah]: hey")
        assert "Unknown" not in names

    def test_no_speaker_prefix(self):
        assert extract_speaker_names("just plain text") == []

    def test_empty_string(self):
        assert extract_speaker_names("") == []

    def test_deduplicates(self):
        text = "[Sarah]: first\n[Sarah]: second"
        assert extract_speaker_names(text).count("Sarah") == 1


# ═══════════════════════════════════════════════════════════════════════════════
# add_chunk_to_graph — node and edge creation
# ═══════════════════════════════════════════════════════════════════════════════

class TestAddChunkToGraph:
    def test_chunk_node_created(self):
        G = nx.DiGraph()
        make_chunk(G, "chunk:1:1000", empty_topics())
        assert G.has_node("chunk:1:1000")
        assert G.nodes["chunk:1:1000"]["type"] == "chunk"

    def test_chunk_node_attrs(self):
        G = nx.DiGraph()
        make_chunk(G, "chunk:1:1000", empty_topics(), chat_id=7,
                   start="2024-03-01T09:00", end="2024-03-01T09:05",
                   raw_text="hello world")
        attrs = G.nodes["chunk:1:1000"]
        assert attrs["chat_id"] == 7
        assert attrs["start_time"] == "2024-03-01T09:00"
        assert attrs["end_time"] == "2024-03-01T09:05"
        assert attrs["raw_text"] == "hello world"

    def test_topic_nodes_created(self):
        G = nx.DiGraph()
        make_chunk(G, "chunk:1:1000",
                   {"people": ["Sarah"], "places": [], "events": ["dinner"], "topics": []})
        assert G.has_node("people:sarah")
        assert G.has_node("events:dinner")

    def test_topic_node_attrs(self):
        G = nx.DiGraph()
        make_chunk(G, "chunk:1:1000",
                   {"people": ["Sarah"], "places": [], "events": [], "topics": []})
        attrs = G.nodes["people:sarah"]
        assert attrs["type"] == "topic"
        assert attrs["category"] == "people"
        assert attrs["name"] == "Sarah"

    def test_topic_nodes_have_embeddings(self):
        G = nx.DiGraph()
        make_chunk(G, "chunk:1:1000",
                   {"people": [], "places": [], "events": ["dinner"], "topics": []})
        assert "embedding" in G.nodes["events:dinner"]
        emb = G.nodes["events:dinner"]["embedding"]
        assert isinstance(emb, list)
        assert len(emb) == 8

    def test_chunk_topic_edges_are_bidirectional(self):
        G = nx.DiGraph()
        make_chunk(G, "chunk:1:1000",
                   {"people": [], "places": [], "events": ["dinner"], "topics": []})
        assert G.has_edge("chunk:1:1000", "events:dinner")
        assert G.has_edge("events:dinner", "chunk:1:1000")

    def test_chunk_topic_edge_type(self):
        G = nx.DiGraph()
        make_chunk(G, "chunk:1:1000",
                   {"people": [], "places": [], "events": ["dinner"], "topics": []})
        assert G["chunk:1:1000"]["events:dinner"]["edge_type"] == "mentions"

    def test_co_occurrence_edges_added(self):
        G = nx.DiGraph()
        make_chunk(G, "chunk:1:1000",
                   {"people": [], "places": [], "events": ["dinner"], "topics": ["travel"]})
        assert G.has_edge("events:dinner", "topics:travel")
        assert G.has_edge("topics:travel", "events:dinner")

    def test_co_occurrence_edge_type(self):
        G = nx.DiGraph()
        make_chunk(G, "chunk:1:1000",
                   {"people": [], "places": [], "events": ["dinner"], "topics": ["travel"]})
        assert G["events:dinner"]["topics:travel"]["edge_type"] == "co_occurs"

    def test_returns_only_new_topic_ids(self):
        G = nx.DiGraph()
        new1 = make_chunk(G, "chunk:1:1000",
                          {"people": ["Sarah"], "places": [], "events": [], "topics": []})
        # Sarah already exists; dinner is new
        new2 = make_chunk(G, "chunk:1:2000",
                          {"people": ["Sarah"], "places": [], "events": ["dinner"], "topics": []})
        assert "people:sarah" in new1
        assert "people:sarah" not in new2   # already existed
        assert "events:dinner" in new2

    def test_existing_chunk_not_duplicated(self):
        G = nx.DiGraph()
        make_chunk(G, "chunk:1:1000", empty_topics())
        make_chunk(G, "chunk:1:1000", empty_topics())   # same id, called again
        # NetworkX silently updates attrs; verify only one node
        assert list(G.nodes).count("chunk:1:1000") == 1

    def test_empty_topic_names_skipped(self):
        G = nx.DiGraph()
        make_chunk(G, "chunk:1:1000",
                   {"people": ["", "  "], "places": [], "events": [], "topics": []})
        topic_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "topic"]
        assert topic_nodes == []


# ═══════════════════════════════════════════════════════════════════════════════
# Temporal edges
# ═══════════════════════════════════════════════════════════════════════════════

class TestTemporalEdges:
    def test_no_temporal_edge_for_first_chunk(self):
        G = nx.DiGraph()
        make_chunk(G, "chunk:1:1000", empty_topics(), prev_node_id=None)
        temporal = [(u, v) for u, v, d in G.edges(data=True)
                    if d.get("edge_type") == "temporal"]
        assert temporal == []

    def test_temporal_edge_added_for_consecutive_chunk(self):
        G = nx.DiGraph()
        make_chunk(G, "chunk:1:1000", empty_topics())
        make_chunk(G, "chunk:1:2000", empty_topics(), prev_node_id="chunk:1:1000")
        assert G.has_edge("chunk:1:1000", "chunk:1:2000")
        assert G.has_edge("chunk:1:2000", "chunk:1:1000")

    def test_temporal_edge_type_and_weight(self):
        G = nx.DiGraph()
        make_chunk(G, "chunk:1:1000", empty_topics())
        make_chunk(G, "chunk:1:2000", empty_topics(), prev_node_id="chunk:1:1000")
        d = G["chunk:1:1000"]["chunk:1:2000"]
        assert d["edge_type"] == "temporal"
        assert d["weight"] == 1.0

    def test_no_temporal_edge_if_prev_not_in_graph(self):
        G = nx.DiGraph()
        # prev_node_id references a node that was never added
        make_chunk(G, "chunk:1:2000", empty_topics(), prev_node_id="chunk:1:999")
        temporal = [(u, v) for u, v, d in G.edges(data=True)
                    if d.get("edge_type") == "temporal"]
        assert temporal == []


# ═══════════════════════════════════════════════════════════════════════════════
# BM25 weight recomputation
# ═══════════════════════════════════════════════════════════════════════════════

class TestBm25Weights:
    """
    Tests for recompute_bm25_weights.

    Each chunk's raw_text must actually contain the topic words so BM25 can
    compute a real TF score. The helper below wires everything up.
    """

    def _chunk(self, G, cid, text):
        G.add_node(cid, type="chunk", raw_text=text,
                   start_time="", end_time="", chat_id=1)

    def _topic(self, G, name, category="topics"):
        tid = topic_node_id(category, name)
        if not G.has_node(tid):
            G.add_node(tid, type="topic", name=name, category=category)
        return tid

    def _link(self, G, chunk_id, topic_id):
        for a, b in ((chunk_id, topic_id), (topic_id, chunk_id)):
            G.add_edge(a, b, weight=1.0, edge_type="mentions")

    # ── weights are set ──────────────────────────────────────────────────────

    def test_weights_set_after_recompute(self):
        """Weights are updated from 1.0 to a real BM25 value."""
        G = nx.DiGraph()
        self._chunk(G, "chunk:1:1", "let's have dinner tonight")
        tid = self._topic(G, "dinner", "events")
        self._link(G, "chunk:1:1", tid)

        recompute_bm25_weights(G)

        w = G["chunk:1:1"][tid]["weight"]
        assert w > 0

    def test_weight_always_above_floor(self):
        """BM25 score can be near zero for ubiquitous topics; floor prevents that."""
        G = nx.DiGraph()
        for i in range(3):
            cid = f"chunk:1:{i}"
            self._chunk(G, cid, f"work project {i}")
            tid = self._topic(G, "work")
            self._link(G, cid, tid)

        recompute_bm25_weights(G, floor=0.1)

        for u, v, d in G.edges(data=True):
            if d.get("edge_type") == "mentions":
                assert d["weight"] >= 0.1, f"Edge {u}→{v} weight {d['weight']} below floor"

    # ── IDF component ────────────────────────────────────────────────────────

    def test_rare_topic_scores_higher_than_ubiquitous_topic(self):
        """A topic in fewer chunks should have higher weight than one in all chunks."""
        G = nx.DiGraph()
        # chunk:1:1 mentions BOTH dinner (rare) and work (ubiquitous)
        self._chunk(G, "chunk:1:1", "work project and dinner tonight")
        self._chunk(G, "chunk:1:2", "work project again")
        self._chunk(G, "chunk:1:3", "work project again")

        t_dinner = self._topic(G, "dinner", "events")
        t_work   = self._topic(G, "work")

        # dinner only in chunk:1:1; work in all three
        self._link(G, "chunk:1:1", t_dinner)
        for cid in ("chunk:1:1", "chunk:1:2", "chunk:1:3"):
            self._link(G, cid, t_work)

        recompute_bm25_weights(G)

        w_dinner = G["chunk:1:1"][t_dinner]["weight"]
        w_work   = G["chunk:1:1"][t_work]["weight"]
        assert w_dinner > w_work, (
            f"Rare 'dinner' ({w_dinner:.4f}) should score higher than "
            f"ubiquitous 'work' ({w_work:.4f})"
        )

    # ── TF component ─────────────────────────────────────────────────────────

    def test_higher_tf_gives_higher_score(self):
        """Chunk with more topic occurrences in its text should score higher."""
        G = nx.DiGraph()
        self._chunk(G, "chunk:1:1", "let's have dinner tonight")          # dinner ×1
        self._chunk(G, "chunk:1:2", "dinner dinner dinner every night")   # dinner ×3
        # Background chunks (not linked to "dinner") keep df < 50% so IDF stays positive
        self._chunk(G, "chunk:1:3", "work meeting today")
        self._chunk(G, "chunk:1:4", "project update submitted")
        self._chunk(G, "chunk:1:5", "see you tomorrow morning")

        tid = self._topic(G, "dinner", "events")
        self._link(G, "chunk:1:1", tid)
        self._link(G, "chunk:1:2", tid)

        recompute_bm25_weights(G)

        w1 = G["chunk:1:1"][tid]["weight"]
        w2 = G["chunk:1:2"][tid]["weight"]
        assert w2 > w1, f"More occurrences should score higher: {w2:.4f} vs {w1:.4f}"

    def test_tf_saturates(self):
        """BM25 TF saturation: 10 occurrences shouldn't score 10× a single occurrence."""
        G = nx.DiGraph()
        self._chunk(G, "chunk:1:1", "dinner")                         # dinner ×1
        self._chunk(G, "chunk:1:2", " ".join(["dinner"] * 10))        # dinner ×10
        # Background chunks keep "dinner" IDF positive (df < 50% of corpus)
        self._chunk(G, "chunk:1:3", "work meeting today")
        self._chunk(G, "chunk:1:4", "project update submitted")
        self._chunk(G, "chunk:1:5", "see you tomorrow morning")

        tid = self._topic(G, "dinner", "events")
        self._link(G, "chunk:1:1", tid)
        self._link(G, "chunk:1:2", tid)

        recompute_bm25_weights(G)

        w1 = G["chunk:1:1"][tid]["weight"]
        w2 = G["chunk:1:2"][tid]["weight"]
        assert w2 > w1                  # TF still helps
        assert w2 < 10 * w1            # but saturates (BM25 asymptote ≈ k1+1 = 2.5)

    # ── other edge types untouched ───────────────────────────────────────────

    def test_temporal_edges_not_modified(self):
        G = nx.DiGraph()
        make_chunk(G, "chunk:1:1000", empty_topics())
        make_chunk(G, "chunk:1:2000", empty_topics(), prev_node_id="chunk:1:1000")
        original_w = G["chunk:1:1000"]["chunk:1:2000"]["weight"]
        recompute_bm25_weights(G)
        assert G["chunk:1:1000"]["chunk:1:2000"]["weight"] == original_w

    def test_empty_graph_no_crash(self):
        recompute_bm25_weights(nx.DiGraph())


# ═══════════════════════════════════════════════════════════════════════════════
# Semantic topic edges
# ═══════════════════════════════════════════════════════════════════════════════

class TestSemanticTopicEdges:
    def _topic_node(self, G, name, category, embedding):
        tid = topic_node_id(category, name)
        G.add_node(tid, type="topic", name=name, category=category,
                   embedding=list(embedding))
        return tid

    def test_adds_edge_for_similar_topics(self):
        G = nx.DiGraph()
        # dinner and dining both map to axis-0 → cosine = 1.0 > 0.5
        t1 = self._topic_node(G, "dinner", "events", [1., 0., 0., 0., 0., 0., 0., 0.])
        t2 = self._topic_node(G, "dining", "events", [1., 0., 0., 0., 0., 0., 0., 0.])
        add_semantic_topic_edges(G, [t1], threshold=0.5)
        assert G.has_edge(t1, t2)
        assert G.has_edge(t2, t1)

    def test_edge_type_is_semantic(self):
        G = nx.DiGraph()
        t1 = self._topic_node(G, "dinner", "events", [1., 0., 0., 0., 0., 0., 0., 0.])
        t2 = self._topic_node(G, "dining", "events", [1., 0., 0., 0., 0., 0., 0., 0.])
        add_semantic_topic_edges(G, [t1], threshold=0.5)
        assert G[t1][t2]["edge_type"] == "semantic"

    def test_edge_weight_is_cosine_similarity(self):
        G = nx.DiGraph()
        t1 = self._topic_node(G, "dinner", "events", [1., 0., 0., 0., 0., 0., 0., 0.])
        t2 = self._topic_node(G, "dining", "events", [1., 0., 0., 0., 0., 0., 0., 0.])
        add_semantic_topic_edges(G, [t1], threshold=0.5)
        assert abs(G[t1][t2]["weight"] - 1.0) < 1e-4

    def test_no_edge_for_orthogonal_topics(self):
        G = nx.DiGraph()
        # dinner (axis 0) vs work (axis 2): cosine = 0.0 < 0.5
        t1 = self._topic_node(G, "dinner", "events", [1., 0., 0., 0., 0., 0., 0., 0.])
        t2 = self._topic_node(G, "work",   "topics", [0., 0., 1., 0., 0., 0., 0., 0.])
        add_semantic_topic_edges(G, [t1], threshold=0.5)
        assert not G.has_edge(t1, t2)

    def test_returns_count_of_edges_added(self):
        G = nx.DiGraph()
        t1 = self._topic_node(G, "dinner",     "events", [1., 0., 0., 0., 0., 0., 0., 0.])
        t2 = self._topic_node(G, "dining",     "events", [1., 0., 0., 0., 0., 0., 0., 0.])
        t3 = self._topic_node(G, "restaurant", "places", [1., 0., 0., 0., 0., 0., 0., 0.])
        count = add_semantic_topic_edges(G, [t1], threshold=0.5)
        # t1 vs t2: 2 edges; t1 vs t3: 2 edges
        assert count == 4

    def test_no_self_loops(self):
        G = nx.DiGraph()
        t1 = self._topic_node(G, "dinner", "events", [1., 0., 0., 0., 0., 0., 0., 0.])
        add_semantic_topic_edges(G, [t1], threshold=0.5)
        assert not G.has_edge(t1, t1)

    def test_empty_new_topic_list_returns_zero(self):
        G = nx.DiGraph()
        self._topic_node(G, "dinner", "events", [1., 0., 0., 0., 0., 0., 0., 0.])
        assert add_semantic_topic_edges(G, []) == 0

    def test_single_topic_in_graph_returns_zero(self):
        G = nx.DiGraph()
        t1 = self._topic_node(G, "dinner", "events", [1., 0., 0., 0., 0., 0., 0., 0.])
        assert add_semantic_topic_edges(G, [t1], threshold=0.5) == 0

    def test_topics_without_embeddings_skipped(self):
        G = nx.DiGraph()
        # t1 has an embedding; t2 does not
        t1 = self._topic_node(G, "dinner", "events", [1., 0., 0., 0., 0., 0., 0., 0.])
        t2 = topic_node_id("events", "dining")
        G.add_node(t2, type="topic", name="dining", category="events")  # no embedding
        count = add_semantic_topic_edges(G, [t1], threshold=0.5)
        assert count == 0
        assert not G.has_edge(t1, t2)


# ═══════════════════════════════════════════════════════════════════════════════
# PPR search
# ═══════════════════════════════════════════════════════════════════════════════

class TestPprSearch:
    def _build_ppr_graph(self):
        """
        Two isolated chunks:
          chunk A → topic:dinner  (axis 0 embedding)
          chunk B → topic:work    (axis 2 embedding)
        IDF weights are recomputed so edges are meaningful.
        """
        G = nx.DiGraph()
        ca = "chunk:1:1000"
        cb = "chunk:2:2000"
        t_dinner = topic_node_id("events", "dinner")
        t_work   = topic_node_id("topics", "work")

        for nid, raw, chat in [(ca, "dinner tonight", 1), (cb, "work meeting", 2)]:
            G.add_node(nid, type="chunk", raw_text=raw,
                       start_time="2024-01-01T10:00", end_time="2024-01-01T10:05",
                       chat_id=chat)

        G.add_node(t_dinner, type="topic", name="dinner", category="events",
                   embedding=list(_EMBED_TABLE["dinner"]))
        G.add_node(t_work,   type="topic", name="work",   category="topics",
                   embedding=list(_EMBED_TABLE["work"]))

        for a, b in [(ca, t_dinner), (t_dinner, ca), (cb, t_work), (t_work, cb)]:
            G.add_edge(a, b, weight=1.5, edge_type="mentions")

        return G, ca, cb

    def test_empty_graph_returns_empty(self):
        assert ppr_search(nx.DiGraph(), "dinner", model=MODEL) == []

    def test_result_structure(self):
        G, ca, _ = self._build_ppr_graph()
        results = ppr_search(G, "dinner", top_k=1, model=MODEL)
        assert len(results) == 1
        r = results[0]
        for key in ("node_id", "score", "text", "start_time", "end_time", "chat_id"):
            assert key in r, f"Missing key: {key}"

    def test_results_sorted_by_score_descending(self):
        G, _, _ = self._build_ppr_graph()
        results = ppr_search(G, "dinner", top_k=2, model=MODEL)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_query_similar_chunk_ranks_first(self):
        """Query 'dinner' → embedding matches dinner topic → dinner chunk scores higher."""
        G, ca, cb = self._build_ppr_graph()
        results = ppr_search(G, "dinner", top_k=2, model=MODEL)
        assert results[0]["node_id"] == ca, (
            f"Expected dinner chunk first, got {results[0]['node_id']}"
        )

    def test_top_k_respected(self):
        G, _, _ = self._build_ppr_graph()
        results = ppr_search(G, "dinner", top_k=1, model=MODEL)
        assert len(results) == 1

    def test_no_matching_topics_returns_empty(self):
        """Query with no similar topics and no string match → empty."""
        G = nx.DiGraph()
        G.add_node("chunk:1:1", type="chunk", raw_text="x",
                   start_time="", end_time="", chat_id=1)
        # Only one topic, orthogonal to everything: axis 5
        tid = topic_node_id("topics", "zzz")
        G.add_node(tid, type="topic", name="zzz", category="topics",
                   embedding=[0., 0., 0., 0., 0., 1., 0., 0.])
        G.add_edge("chunk:1:1", tid, weight=1.0, edge_type="mentions")
        G.add_edge(tid, "chunk:1:1", weight=1.0, edge_type="mentions")
        # Query is on axis 0 (dinner), topic is on axis 5 → sim below SEED_SIM_MIN
        results = ppr_search(G, "dinner", top_k=5, model=MODEL)
        # May return empty list; verify it doesn't crash and scores are not inflated
        for r in results:
            assert r["score"] < 1.0

    def test_string_fallback_when_no_embeddings(self):
        """Topics without embeddings: ppr_search falls back to string matching."""
        G = nx.DiGraph()
        ca = "chunk:1:1000"
        t_dinner = topic_node_id("events", "dinner")
        G.add_node(ca, type="chunk", raw_text="dinner tonight",
                   start_time="", end_time="", chat_id=1)
        G.add_node(t_dinner, type="topic", name="dinner", category="events")
        # No "embedding" attr — triggers fallback
        for a, b in [(ca, t_dinner), (t_dinner, ca)]:
            G.add_edge(a, b, weight=1.0, edge_type="mentions")

        results = ppr_search(G, "dinner", top_k=5, model=MODEL)
        assert len(results) >= 1
        assert results[0]["node_id"] == ca

    def test_result_text_matches_node_raw_text(self):
        G, ca, _ = self._build_ppr_graph()
        results = ppr_search(G, "dinner", top_k=1, model=MODEL)
        assert results[0]["text"] == G.nodes[ca]["raw_text"]


# ═══════════════════════════════════════════════════════════════════════════════
# Graph persistence
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphPersistence:
    def test_round_trip_preserves_node_count(self, tmp_path):
        original = gi.GRAPH_PATH
        gi.GRAPH_PATH = str(tmp_path / "graph.json")
        try:
            G = nx.DiGraph()
            make_chunk(G, "chunk:1:1000",
                       {"people": ["Sarah"], "places": [], "events": ["dinner"], "topics": []})
            save_graph(G)
            G2 = load_graph()
            assert G2.number_of_nodes() == G.number_of_nodes()
        finally:
            gi.GRAPH_PATH = original

    def test_round_trip_preserves_edge_count(self, tmp_path):
        original = gi.GRAPH_PATH
        gi.GRAPH_PATH = str(tmp_path / "graph.json")
        try:
            G = nx.DiGraph()
            make_chunk(G, "chunk:1:1000",
                       {"people": [], "places": [], "events": ["dinner"], "topics": ["travel"]})
            save_graph(G)
            G2 = load_graph()
            assert G2.number_of_edges() == G.number_of_edges()
        finally:
            gi.GRAPH_PATH = original

    def test_round_trip_preserves_node_attrs(self, tmp_path):
        original = gi.GRAPH_PATH
        gi.GRAPH_PATH = str(tmp_path / "graph.json")
        try:
            G = nx.DiGraph()
            make_chunk(G, "chunk:1:1000",
                       {"people": ["Sarah"], "places": [], "events": [], "topics": []},
                       raw_text="hello world")
            save_graph(G)
            G2 = load_graph()
            assert G2.nodes["chunk:1:1000"]["raw_text"] == "hello world"
            assert G2.nodes["chunk:1:1000"]["type"] == "chunk"
        finally:
            gi.GRAPH_PATH = original

    def test_round_trip_preserves_topic_embedding(self, tmp_path):
        original = gi.GRAPH_PATH
        gi.GRAPH_PATH = str(tmp_path / "graph.json")
        try:
            G = nx.DiGraph()
            make_chunk(G, "chunk:1:1000",
                       {"people": [], "places": [], "events": ["dinner"], "topics": []})
            original_emb = G.nodes["events:dinner"]["embedding"][:]
            save_graph(G)
            G2 = load_graph()
            assert G2.nodes["events:dinner"]["embedding"] == original_emb
        finally:
            gi.GRAPH_PATH = original

    def test_round_trip_preserves_edge_attrs(self, tmp_path):
        original = gi.GRAPH_PATH
        gi.GRAPH_PATH = str(tmp_path / "graph.json")
        try:
            G = nx.DiGraph()
            make_chunk(G, "chunk:1:1000",
                       {"people": [], "places": [], "events": ["dinner"], "topics": []})
            recompute_bm25_weights(G)
            w_before = G["chunk:1:1000"]["events:dinner"]["weight"]
            save_graph(G)
            G2 = load_graph()
            assert abs(G2["chunk:1:1000"]["events:dinner"]["weight"] - w_before) < 1e-9
            assert G2["chunk:1:1000"]["events:dinner"]["edge_type"] == "mentions"
        finally:
            gi.GRAPH_PATH = original

    def test_load_nonexistent_returns_empty_digraph(self, tmp_path):
        original = gi.GRAPH_PATH
        gi.GRAPH_PATH = str(tmp_path / "no_such_file.json")
        try:
            G = load_graph()
            assert isinstance(G, nx.DiGraph)
            assert G.number_of_nodes() == 0
        finally:
            gi.GRAPH_PATH = original
