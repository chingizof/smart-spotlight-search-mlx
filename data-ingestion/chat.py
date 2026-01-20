import ollama
from search import search

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

Use ONLY the following messages as context. If the answer isn't in the context, say "I couldn't find that in your messages."

Context:
{context}

Question: {query}

Answer concisely based on the messages above."""


# --- 3. Format search results as context ---
def get_context(query: str) -> str:
    results = search(query)
    if results.empty:
        return "No relevant messages found."

    context_lines = []
    for _, row in results.iterrows():
        context_lines.append(f"[{row['timestamp']}] {row['sender_name']}: {row['text']}")

    return "\n".join(context_lines)


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
