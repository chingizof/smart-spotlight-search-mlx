import sqlite3
import Contacts
import re
import datetime
from sentence_transformers import SentenceTransformer

# --- 1. Build Contact Map ---
contact_map = {}
store = Contacts.CNContactStore.alloc().init()
keys = [Contacts.CNContactGivenNameKey, Contacts.CNContactFamilyNameKey, 
        Contacts.CNContactPhoneNumbersKey, Contacts.CNContactEmailAddressesKey]
request = Contacts.CNContactFetchRequest.alloc().initWithKeysToFetch_(keys)

def process_contact(c, stop):
    name = f"{c.givenName()} {c.familyName()}".strip() or "Unknown"
    for p in c.phoneNumbers():
        digits = re.sub(r'\D', '', p.value().stringValue())
        if len(digits) >= 10:
            contact_map[digits[-10:]] = name
    for e in c.emailAddresses():
        contact_map[e.value().lower()] = name

store.enumerateContactsWithFetchRequest_error_usingBlock_(request, None, process_contact)

# --- 2. Main Logic ---
DB = "chat-history/chat.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

q = """
SELECT
  m.ROWID,
  h.id AS contact_handle,
  m.text,
  m.date
FROM message m
LEFT JOIN handle h ON m.handle_id = h.ROWID
WHERE m.text IS NOT NULL
ORDER BY m.date ASC
"""

# Extract 1 message for demonstration
row = cur.execute(q).fetchone()
message_id = row[0]
handle_id = row[1]
text = row[2]
raw_date = row[3]

# Resolve Name
if handle_id:
    key = re.sub(r'\D', '', handle_id)[-10:] if "@" not in handle_id else handle_id.lower()
    name = contact_map.get(key, handle_id)
    # Check for raw phone number (+ and 11+ digits) -> unknown
    if re.match(r'^\+\d{11,}$', name):
        name = 'unknown'
else:
    name = "Me"

# Convert Timestamp (iMessage uses nanoseconds since 2001-01-01)
# 978307200 is the Unix offset for 2001-01-01
unix_timestamp = (raw_date / 1_000_000_000) + 978307200
readable_date = datetime.datetime.fromtimestamp(unix_timestamp).isoformat()

# Create Embedding
content_to_embed = f"{name}: {text}"
vector = model.encode(f"search_query: {content_to_embed}")

# --- 3. Final Data Structure ---
data_row = {
    "vector": vector.tolist(),       # 768d embedding
    "text": text,                    # Raw text
    "timestamp": readable_date,      # ISO format for temporal search
    "sender_name": name,             # Resolved name or 'Me'
    "message_id": message_id         # Original DB ID
}

print("Data Row Created:")
print(f"ID: {data_row['message_id']}")
print(f"Date: {data_row['timestamp']}")
print(f"Sender: {data_row['sender_name']}")
print(f"Text: {data_row['text']}")
print(f"Vector (len): {len(data_row['vector'])}")

conn.close()