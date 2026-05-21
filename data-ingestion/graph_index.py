"""
Knowledge graph index over iMessage chunks.

Graph schema
------------
Nodes
  type=chunk  id="chunk:{chat_id}:{unix_ts_int}"
              attrs: raw_text, start_time, end_time, chat_id

  type=topic  id="{category}:{name_lower}"
              attrs: name (original casing), category,
                     embedding (list[float], stored as plain list for JSON)

Edges (all bidirectional)
  chunk ↔ topic   edge_type="mentions"   weight=IDF(topic)  [set by recompute_idf_weights]
  topic ↔ topic   edge_type="co_occurs"  weight=co-occurrence count
  topic ↔ topic   edge_type="semantic"   weight=cosine_similarity  [added by add_semantic_topic_edges]
  chunk ↔ chunk   edge_type="temporal"   weight=1

Retrieval
---------
ppr_search(G, query, model?)
  → embed query with same nomic model
  → rank all topic nodes by cosine(query, topic_embedding)
  → seed PPR from top-K similar topics
  → return top-N chunk nodes by PPR score

IDF weighting
-------------
recompute_idf_weights(G)   call once after a batch of add_chunk_to_graph calls.
  weight(chunk↔topic) = log((total_chunks + 1) / (chunks_mentioning_topic + 1))

Semantic topic edges
--------------------
add_semantic_topic_edges(G, new_topic_ids, threshold=0.5)
  For each newly added topic, compute cosine against all existing topic embeddings.
  Add a "semantic" edge for any pair above the threshold.
"""

import json
import math
import re
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

GRAPH_PATH = "graph.json"
EMBED_MODEL_NAME = "nomic-ai/nomic-embed-text-v1"
EMBED_PREFIX = "search_query: "

TOPIC_SIM_THRESHOLD = 0.50  # min cosine to create a semantic topic↔topic edge
SEED_SIM_MIN = 0.25         # min cosine for a topic to be used as a PPR seed
SEED_TOP_K = 10             # how many top topics to seed PPR from

_embed_model = None


def _get_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME, trust_remote_code=True)
    return _embed_model


def _embed(texts: list[str] | str, model=None) -> np.ndarray:
    """Embed one or more strings. Always returns a 2-D array (n, dim)."""
    if model is None:
        model = _get_model()
    if isinstance(texts, str):
        texts = [texts]
    prefixed = [EMBED_PREFIX + t for t in texts]
    vecs = model.encode(prefixed, convert_to_numpy=True)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / (norms + 1e-8)   # return L2-normalised so dot == cosine


# ── Persistence ───────────────────────────────────────────────────────────────

def load_graph() -> nx.DiGraph:
    path = Path(GRAPH_PATH)
    if path.exists():
        data = json.loads(path.read_text())
        return nx.node_link_graph(data, directed=True, multigraph=False)
    return nx.DiGraph()


def save_graph(G: nx.DiGraph) -> None:
    Path(GRAPH_PATH).write_text(json.dumps(nx.node_link_data(G)))


# ── Node ID helpers ───────────────────────────────────────────────────────────

def chunk_node_id(chat_id: int, unix_ts: float) -> str:
    return f"chunk:{chat_id}:{int(unix_ts)}"


def topic_node_id(category: str, name: str) -> str:
    return f"{category}:{name.strip().lower()}"


# ── Graph building ────────────────────────────────────────────────────────────

def add_chunk_to_graph(
    G: nx.DiGraph,
    node_id: str,
    raw_text: str,
    topics: dict[str, list[str]],
    start_time: str,
    end_time: str,
    chat_id: int,
    model,
    prev_node_id: Optional[str] = None,
) -> list[str]:
    """
    Add a chunk node and its topic nodes/edges to G.

    Returns the list of topic node IDs that were newly created in this call
    (not previously in the graph). Pass this list to add_semantic_topic_edges
    and recompute_idf_weights after processing a batch.
    """
    G.add_node(
        node_id,
        type="chunk",
        raw_text=raw_text,
        start_time=start_time,
        end_time=end_time,
        chat_id=chat_id,
    )

    if prev_node_id and G.has_node(prev_node_id):
        for a, b in ((prev_node_id, node_id), (node_id, prev_node_id)):
            if not G.has_edge(a, b):
                G.add_edge(a, b, weight=1.0, edge_type="temporal")

    # Gather all topic names for batch embedding (only new ones)
    new_topic_ids: list[str] = []
    new_topic_names: list[str] = []
    chunk_topic_ids: list[str] = []

    for category, names in topics.items():
        for name in names:
            if not name or not name.strip():
                continue
            tid = topic_node_id(category, name)
            chunk_topic_ids.append(tid)

            if not G.has_node(tid):
                G.add_node(tid, type="topic", name=name, category=category)
                new_topic_ids.append(tid)
                new_topic_names.append(name)

    # Embed all new topic names in one batch call
    if new_topic_names:
        embeddings = _embed(new_topic_names, model=model)
        for tid, vec in zip(new_topic_ids, embeddings):
            G.nodes[tid]["embedding"] = vec.tolist()

    # chunk ↔ topic edges (weight=1 for now; IDF pass sets final weights)
    for tid in chunk_topic_ids:
        for a, b in ((node_id, tid), (tid, node_id)):
            if G.has_edge(a, b):
                G[a][b]["weight"] = G[a][b].get("weight", 1.0) + 1.0
            else:
                G.add_edge(a, b, weight=1.0, edge_type="mentions")

    # topic ↔ topic co-occurrence edges
    for i, t1 in enumerate(chunk_topic_ids):
        for t2 in chunk_topic_ids[i + 1:]:
            if t1 == t2:
                continue
            for a, b in ((t1, t2), (t2, t1)):
                if G.has_edge(a, b):
                    G[a][b]["weight"] = G[a][b].get("weight", 1.0) + 1.0
                else:
                    G.add_edge(a, b, weight=1.0, edge_type="co_occurs")

    return new_topic_ids


def add_semantic_topic_edges(
    G: nx.DiGraph,
    new_topic_ids: list[str],
    threshold: float = TOPIC_SIM_THRESHOLD,
) -> int:
    """
    For each newly added topic, compute cosine similarity against all existing
    topic embeddings and add a "semantic" edge for pairs above the threshold.

    Uses batch matrix multiply so it's fast even for 10k topics.
    Returns number of new semantic edges added.
    """
    if not new_topic_ids:
        return 0

    # Collect all topic nodes that have embeddings
    all_topic_ids = [
        n for n, d in G.nodes(data=True)
        if d.get("type") == "topic" and "embedding" in d
    ]
    if len(all_topic_ids) < 2:
        return 0

    # Stack into matrix (already L2-normalised from _embed)
    all_vecs = np.array([G.nodes[n]["embedding"] for n in all_topic_ids])  # (M, dim)

    # Index positions of the new topics within all_topic_ids
    all_ids_set = {n: i for i, n in enumerate(all_topic_ids)}
    new_indices = [all_ids_set[tid] for tid in new_topic_ids if tid in all_ids_set]

    if not new_indices:
        return 0

    new_vecs = all_vecs[new_indices]          # (K, dim)
    sim_matrix = new_vecs @ all_vecs.T        # (K, M)  — cosine because normalised

    added = 0
    for k, global_i in enumerate(new_indices):
        new_tid = all_topic_ids[global_i]
        sims = sim_matrix[k]                  # shape (M,)

        for m, sim in enumerate(sims):
            if m == global_i:                 # skip self
                continue
            if sim < threshold:
                continue
            other_tid = all_topic_ids[m]

            for a, b in ((new_tid, other_tid), (other_tid, new_tid)):
                if not G.has_edge(a, b):
                    G.add_edge(a, b, weight=float(sim), edge_type="semantic")
                    added += 1
                elif G[a][b].get("edge_type") != "mentions":
                    # Upgrade weight if this semantic link is stronger
                    G[a][b]["weight"] = max(G[a][b]["weight"], float(sim))

    return added


def recompute_idf_weights(G: nx.DiGraph) -> None:
    """
    Set chunk↔topic edge weights to IDF scores.

    IDF(topic) = log((total_chunks + 1) / (chunks_mentioning_topic + 1))

    Topics appearing in many chunks get low weight (common signal).
    Topics appearing in few chunks get high weight (specific signal).
    """
    total_chunks = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "chunk")
    if total_chunks == 0:
        return

    for topic_node, attrs in G.nodes(data=True):
        if attrs.get("type") != "topic":
            continue

        chunk_neighbors = [
            n for n in G.neighbors(topic_node)
            if G.nodes[n].get("type") == "chunk"
        ]
        df = len(chunk_neighbors)
        if df == 0:
            continue

        # Smooth IDF: +1 ensures weight >= 1 even when topic appears in every chunk
        idf = math.log((total_chunks + 1) / (df + 1)) + 1.0

        for chunk_node in chunk_neighbors:
            for a, b in ((topic_node, chunk_node), (chunk_node, topic_node)):
                if G.has_edge(a, b) and G[a][b].get("edge_type") == "mentions":
                    G[a][b]["weight"] = idf


def extract_speaker_names(raw_text: str) -> list[str]:
    """Pull speaker names from formatted conversation text. No LLM needed."""
    names: set[str] = set()
    for line in raw_text.splitlines():
        m = re.match(r"^\[([^\]]+)\]:", line)
        if m:
            name = m.group(1).strip()
            if name and name not in ("Me", "Unknown"):
                names.add(name)
    return list(names)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def _find_seeds_by_string(G: nx.DiGraph, query: str) -> list[str]:
    """Fallback: string-match topic names against the query."""
    q = query.lower()
    return [
        n for n, d in G.nodes(data=True)
        if d.get("type") == "topic"
        and len(d.get("name", "")) >= 3
        and d["name"].lower() in q
    ]


def _find_seeds_by_embedding(
    G: nx.DiGraph,
    query_vec: np.ndarray,
    top_k: int = SEED_TOP_K,
    min_sim: float = SEED_SIM_MIN,
) -> list[tuple[str, float]]:
    """
    Return topic nodes most similar to the query vector, as (node_id, score) pairs.
    query_vec must already be L2-normalised.
    """
    topic_ids = [
        n for n, d in G.nodes(data=True)
        if d.get("type") == "topic" and "embedding" in d
    ]
    if not topic_ids:
        return []

    topic_vecs = np.array([G.nodes[n]["embedding"] for n in topic_ids])  # (M, dim)
    sims = topic_vecs @ query_vec                                          # (M,)

    scored = sorted(
        ((tid, float(s)) for tid, s in zip(topic_ids, sims) if s >= min_sim),
        key=lambda x: x[1],
        reverse=True,
    )
    return scored[:top_k]


def ppr_search(
    G: nx.DiGraph,
    query: str,
    top_k: int = 5,
    alpha: float = 0.85,
    model=None,
) -> list[dict]:
    """
    Embed the query, seed PPR from the most similar topic nodes, and return
    the top-K chunk nodes ranked by PPR score.

    Falls back to string matching if the graph has no topic embeddings yet.
    """
    if G.number_of_nodes() == 0:
        return []

    # Embed query (lazy-load model if not provided)
    query_vec = _embed(query, model=model)[0]   # already L2-normalised, shape (dim,)

    seed_pairs = _find_seeds_by_embedding(G, query_vec)

    # Fallback: string matching (old graphs without embeddings)
    if not seed_pairs:
        string_seeds = _find_seeds_by_string(G, query)
        if not string_seeds:
            return []
        w = 1.0 / len(string_seeds)
        personalization = {s: w for s in string_seeds}
    else:
        # Weight seeds by their cosine similarity to the query
        total = sum(s for _, s in seed_pairs)
        personalization = {n: s / total for n, s in seed_pairs}

    scores: dict[str, float] = nx.pagerank(
        G, alpha=alpha, personalization=personalization, weight="weight"
    )

    chunk_scores = sorted(
        ((n, s) for n, s in scores.items() if G.nodes[n].get("type") == "chunk"),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        {
            "node_id": node,
            "score": round(score, 6),
            "text": G.nodes[node].get("raw_text", ""),
            "start_time": G.nodes[node].get("start_time", ""),
            "end_time": G.nodes[node].get("end_time", ""),
            "chat_id": G.nodes[node].get("chat_id", 0),
        }
        for node, score in chunk_scores[:top_k]
    ]
