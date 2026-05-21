import json

import ollama

MODEL = "llama3.2"

# Prompt tuned for informal conversational text (iMessages).
# JSON mode enforces structure; examples in the rules anchor the model on edge cases.
EXTRACT_PROMPT = """\
Extract topics from this iMessage conversation for a personal search index.

Conversation:
{text}

Return a JSON object with exactly these four keys. No explanation, no markdown:
{{
  "people":  [],
  "places":  [],
  "events":  [],
  "topics":  []
}}

Rules:
- people:  first or full names of people explicitly mentioned or addressed
           e.g. ["Sarah", "Dr. Kim", "Mike"]
           Skip pronouns ("he", "she") and generic labels ("my boss", "the doctor")
           Normalize: if "Mike" and "Michael" both appear, use "Michael"
- places:  specific locations, venues, cities, addresses
           e.g. ["New York", "Blue Bottle Coffee", "LAX", "Mom's house"]
           Skip vague locations ("somewhere", "there", "home" unless clearly meaningful)
- events:  activities, occasions, appointments, plans
           e.g. ["birthday party", "job interview", "flight to Boston", "dinner"]
           Use lowercase 1-4 word phrases
- topics:  high-level themes only — things worth searching by later
           e.g. ["travel", "work", "health", "money", "dating", "sports"]
           Skip obvious/low-signal: "conversation", "message", "chat", "question"
           Max 3 topics per chunk

Only include what is explicitly present. Empty array if nothing fits a category.\
"""


def extract_topics(text: str) -> dict[str, list[str]]:
    """
    Extract structured topics from a conversation chunk via LLM.
    Returns dict with keys: people, places, events, topics.
    Falls back to empty dict on any failure so indexing never hard-stops.
    """
    empty: dict[str, list[str]] = {"people": [], "places": [], "events": [], "topics": []}

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": EXTRACT_PROMPT.format(text=text)}],
            format="json",
        )
        raw = response["message"]["content"]
        parsed = json.loads(raw)
    except Exception:
        return empty

    result: dict[str, list[str]] = {}
    for key in ("people", "places", "events", "topics"):
        items = parsed.get(key, [])
        if not isinstance(items, list):
            result[key] = []
            continue
        cleaned = []
        for item in items:
            s = str(item).strip()
            if s:
                # preserve casing for people/places; lowercase events/topics
                cleaned.append(s if key in ("people", "places") else s.lower())
        result[key] = cleaned

    return result
