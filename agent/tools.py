import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "data-ingestion"))

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
