import sqlite3
from sentence_transformers import SentenceTransformer

DB = "chat-history/chat.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

q = """
SELECT
  m.ROWID,
  h.id AS contact,
  m.text
FROM message m
LEFT JOIN handle h ON m.handle_id = h.ROWID
WHERE m.text IS NOT NULL
ORDER BY m.date ASC
"""

# extract 1 message
row = cur.execute(q).fetchone()
text = row[2]
embedded_row = model.encode(f"search_query: {text}")

print(f'text: + {text}') # text
print(f"embedding: {embedded_row}")

conn.close()
