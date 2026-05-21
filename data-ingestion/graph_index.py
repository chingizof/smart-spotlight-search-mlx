"""
Knowledge graph index over iMessage chunks.

Graph schema
------------
Nodes
  type=chunk  id="chunk:{chat_id}:{unix_ts_int}"
              attrs: raw_text, start_time, end_time, chat_id
  type=topic  id="{category}:{name_lower}"
              attrs: name (original casing), category

Edges (all bidirectional)
  chunk ↔ topic   edge_type="mentions"     weight=1
  topic ↔ topic   edge_type="co_occurs"    weight=N (incremented per shared chunk)
  chunk ↔ chunk   edge_type="temporal"     weight=1 (consecutive in same conversation)

Retrieval
---------
find_seed_nodes(G, query)  → topic nodes whose name appears in query text
ppr_search(G, query)       → run PPR from seeds, return top-K chunk nodes
"""

import json
import re
from pathlib import Path
from typing import Optional

import networkx as nx

GRAPH_PATH = "graph.json"


# ── Persistence ──────────────────────────────────────────────────────────────

def load_graph() -> nx.DiGraph:
    path = Path(GRAPH_PATH)
    if path.exists():
        data = json.loads(path.read_text())
        return nx.node_link_graph(data, directed=True, multigraph=False)
    return nx.DiGraph()


def save_graph(G: nx.DiGraph) -> None:
    data = nx.node_link_data(G)
    Path(GRAPH_PATH).write_text(json.dumps(data))


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
    prev_node_id: Optional[str] = None,
) -> None:
    """Add a single chunk and its topics into the graph."""
    G.add_node(
        node_id,
        type="chunk",
        raw_text=raw_text,
        start_time=start_time,
        end_time=end_time,
        chat_id=chat_id,
    )

    # Temporal edge to the previous chunk in the same conversation
    if prev_node_id and G.has_node(prev_node_id):
        for a, b in ((prev_node_id, node_id), (node_id, prev_node_id)):
            if not G.has_edge(a, b):
                G.add_edge(a, b, weight=1, edge_type="temporal")

    # Collect topic node IDs added in this chunk for co-occurrence edges
    chunk_topic_ids: list[str] = []

    for category, names in topics.items():
        for name in names:
            if not name:
                continue
            tid = topic_node_id(category, name)

            if not G.has_node(tid):
                G.add_node(tid, type="topic", name=name, category=category)

            # Bidirectional chunk ↔ topic edges
            for a, b in ((node_id, tid), (tid, node_id)):
                if G.has_edge(a, b):
                    G[a][b]["weight"] = G[a][b].get("weight", 1) + 1
                else:
                    G.add_edge(a, b, weight=1, edge_type="mentions")

            chunk_topic_ids.append(tid)

    # Topic ↔ topic co-occurrence edges (within this chunk)
    for i, t1 in enumerate(chunk_topic_ids):
        for t2 in chunk_topic_ids[i + 1:]:
            if t1 == t2:
                continue
            for a, b in ((t1, t2), (t2, t1)):
                if G.has_edge(a, b):
                    G[a][b]["weight"] = G[a][b].get("weight", 1) + 1
                else:
                    G.add_edge(a, b, weight=1, edge_type="co_occurs")


def extract_speaker_names(raw_text: str) -> list[str]:
    """
    Pull speaker names from formatted conversation text for free (no LLM).
    Matches lines like "[Me]: ..." or "[Sarah]: ..."
    """
    names: set[str] = set()
    for line in raw_text.splitlines():
        m = re.match(r"^\[([^\]]+)\]:", line)
        if m:
            name = m.group(1).strip()
            if name and name not in ("Me", "Unknown"):
                names.add(name)
    return list(names)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def find_seed_nodes(G: nx.DiGraph, query: str) -> list[str]:
    """
    Find topic nodes whose name appears in the query.
    Matches multi-word names ("New York") and single words (3+ chars).
    """
    query_lower = query.lower()
    seeds: list[str] = []

    for node, attrs in G.nodes(data=True):
        if attrs.get("type") != "topic":
            continue
        name = attrs.get("name", "").lower()
        if not name:
            continue
        # Multi-word topic name present anywhere in query, or
        # single word with length >= 3
        if name in query_lower and (len(name) >= 3):
            seeds.append(node)

    return seeds


def ppr_search(
    G: nx.DiGraph,
    query: str,
    top_k: int = 5,
    alpha: float = 0.85,
) -> list[dict]:
    """
    Run Personalized PageRank seeded from query-matched topic nodes.
    Returns up to top_k chunk dicts sorted by PPR score.
    """
    if G.number_of_nodes() == 0:
        return []

    seeds = find_seed_nodes(G, query)
    if not seeds:
        return []

    weight = 1.0 / len(seeds)
    personalization = {s: weight for s in seeds}

    scores: dict[str, float] = nx.pagerank(
        G, alpha=alpha, personalization=personalization, weight="weight"
    )

    chunk_scores = [
        (node, score)
        for node, score in scores.items()
        if G.nodes[node].get("type") == "chunk"
    ]
    chunk_scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for node, score in chunk_scores[:top_k]:
        attrs = G.nodes[node]
        results.append(
            {
                "node_id": node,
                "score": round(score, 6),
                "text": attrs.get("raw_text", ""),
                "start_time": attrs.get("start_time", ""),
                "end_time": attrs.get("end_time", ""),
                "chat_id": attrs.get("chat_id", 0),
            }
        )

    return results
