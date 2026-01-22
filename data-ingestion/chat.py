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
def get_context(query: str) -> str:
    results = search_memories(query, limit=5)
    if not results:
        return "No relevant messages found."

    context_blocks = []
    for mem in results:
        header = f"[{mem['start_time']} - {mem['end_time']}]"
        context_blocks.append(f"{header}\n{mem['text']}")

    return "\n\n---\n\n".join(context_blocks)


# --- 4. Chat with streaming ---
def chat(query: str):
    context = get_context(query)
    prompt = build_prompt(query, context)

    stream = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}], stream=True)

    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
    print()  # newline after response


# --- 5. Main loop ---
if __name__ == "__main__":
    ensure_model()
    print("\nRAG Chat (type 'quit' to exit)")
    print("-" * 40)

    while True:
        try:
            query = input("\nYou: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue

            print("\nAssistant: ", end="")
            chat(query)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
