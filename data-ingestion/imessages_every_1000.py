# This file reads imessages chat and outputs every one thousands message
# In the future we plan to convert imessage texts into embeddings

import sqlite3

DB = "chat-history/chat.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()

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

i = 0
for rowid, contact, text in cur.execute(q):
    i += 1
    if i % 1000 == 0:
        print(f"\n#{i} contact={contact}")
        print(text)

conn.close()
