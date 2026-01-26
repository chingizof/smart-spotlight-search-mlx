import argparse

import ollama
from imessages import search_memories

# --- 1. Ensure model is available ---
MODEL = "llama3.2"


def ensure_model():
    """Pull model if not available locally."""
    try:
        ollama.show(MODEL)
        print(f"Model '{MODEL}' ready")
    except ollama.ResponseError:
        print(f"Pulling '{MODEL}'... (this may take a few minutes)")
        ollama.pull(MODEL)
        print(f"Model '{MODEL}' ready")


# --- 2. RAG prompt template ---
def build_prompt(query: str, context: str) -> str:
    return f"""You are a helpful assistant answering questions based on the user's personal message history.

Use ONLY the following conversation snippets as context. If the answer isn't in the context, say "I couldn't find that in your messages."

Context:
{context}

Question: {query}

Answer concisely based on the conversations above."""


# --- 3. Format search results as context ---
def get_context(query: str, after: str = None, before: str = None) -> str:
    results = search_memories(query, limit=5, after=after, before=before)
    if not results:
        return "No relevant messages found."

    context_blocks = []
    for mem in results:
        header = f"[{mem['start_time']} - {mem['end_time']}]"
        context_blocks.append(f"{header}\n{mem['text']}")

    return "\n\n---\n\n".join(context_blocks)


# --- 4. Chat with streaming ---
def chat(query: str, after: str = None, before: str = None):
    context = get_context(query, after=after, before=before)
    prompt = build_prompt(query, context)

    stream = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}], stream=True)

    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
    print()  # newline after response


# --- 5. Main loop ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Chat with iMessage history")
    parser.add_argument(
        "--after",
        type=str,
        help="Filter messages after this date. Accepts ISO format (YYYY-MM-DD) or "
        "relative format (e.g., 'last 7 days', 'last week', 'yesterday')",
    )
    parser.add_argument(
        "--before",
        type=str,
        help="Filter messages before this date. Accepts ISO format (YYYY-MM-DD) or "
        "relative format (e.g., 'last 7 days', 'last week', 'yesterday')",
    )
    args = parser.parse_args()

    ensure_model()

    filter_info = []
    if args.after:
        filter_info.append(f"after: {args.after}")
    if args.before:
        filter_info.append(f"before: {args.before}")
    filter_str = f" [{', '.join(filter_info)}]" if filter_info else ""

    print(f"\nRAG Chat{filter_str} (type 'quit' to exit)")
    print("-" * 40)

    while True:
        try:
            query = input("\nYou: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue

            print("\nAssistant: ", end="")
            chat(query, after=args.after, before=args.before)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
