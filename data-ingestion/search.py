import os
import lancedb
from sentence_transformers import SentenceTransformer

# Connect to database (use absolute path relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
db = lancedb.connect(os.path.join(PROJECT_ROOT, "lancedb"))
table = db.open_table("imessages")
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

def search(query: str, limit: int = 5):
    """Search for messages similar to the query."""
    query_vector = model.encode(f"search_query: {query}")
    results = table.search(query_vector).metric("cosine").limit(limit).to_pandas()
    return results

if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "hello"
    print(f"Searching for: '{query}'\n")

    results = search(query)
    for i, row in results.iterrows():
        print(f"[{row['timestamp']}] {row['sender_name']}: {row['text']}")
        print(f"  Score: {row['_distance']:.4f}")
        print()
