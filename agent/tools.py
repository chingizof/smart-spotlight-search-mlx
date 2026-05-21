import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "data-ingestion"))

from graph_index import load_graph, ppr_search
from imessages import search_memories as _search_memories


def search_messages(query: str, after: str = None, before: str = None, limit: int = 5) -> str:
    try:
        results = _search_memories(query, limit=limit, after=after, before=before)
    except Exception as e:
        return f"Search error: {e}"

    if not results:
        return "No relevant messages found."

    parts = []
    for i, mem in enumerate(results, 1):
        header = f"[{mem['start_time']} – {mem['end_time']}] ({mem['message_count']} messages)"
        parts.append(f"Result {i} {header}\n{mem['text']}")

    return "\n\n---\n\n".join(parts)


def graph_search(query: str, top_k: int = 5) -> str:
    """Search via Personalized PageRank on the knowledge graph."""
    try:
        G = load_graph()
    except Exception as e:
        return f"Graph load error: {e}"

    if G.number_of_nodes() == 0:
        return "Knowledge graph is empty. Run: python data-ingestion/imessages.py (or --build-graph-only)"

    results = ppr_search(G, query, top_k=top_k)

    if not results:
        return "No matching topics found in the knowledge graph for this query."

    parts = []
    for i, r in enumerate(results, 1):
        header = f"[{r['start_time'][:16]} – {r['end_time'][11:16]}] (score: {r['score']:.5f})"
        parts.append(f"Graph result {i} {header}\n{r['text']}")

    return "\n\n---\n\n".join(parts)


def grep(pattern: str, path: str = ".") -> str:
    try:
        result = subprocess.run(
            [
                "grep", "-r", "-n", "-i",
                "--include=*.txt", "--include=*.md", "--include=*.py",
                "--include=*.json", "--include=*.csv", "--include=*.yaml",
                pattern, path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout.strip()
        if not output:
            return "No matches found."
        lines = output.split("\n")
        if len(lines) > 30:
            return "\n".join(lines[:30]) + f"\n... ({len(lines) - 30} more matches)"
        return output
    except subprocess.TimeoutExpired:
        return "Grep timed out (10s limit)."
    except FileNotFoundError:
        return "grep not available on this system."
    except Exception as e:
        return f"Grep error: {e}"
