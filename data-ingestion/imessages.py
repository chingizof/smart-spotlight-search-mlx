import sqlite3
import Contacts
import re
import datetime
import lancedb
import pyarrow as pa
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

print(f"Loaded {len(contact_map)} contacts from address book")

# --- 2. Initialize LanceDB ---
db = lancedb.connect("lancedb")

# --- 3. Load Model and Connect to SQLite ---
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

# --- 4. Process All Messages ---
rows = cur.execute(q).fetchall()
print(f"Found {len(rows)} messages to embed")

data = []
batch_size = 100

for i, row in enumerate(rows):
    message_id = row[0]
    handle_id = row[1]
    text = row[2]
    raw_date = row[3]

    # Resolve Name
    if handle_id:
        key = re.sub(r'\D', '', handle_id)[-10:] if "@" not in handle_id else handle_id.lower()
        name = contact_map.get(key, "unknown")
    else:
        name = "Me"

    # Convert Timestamp (iMessage uses nanoseconds since 2001-01-01)
    unix_timestamp = (raw_date / 1_000_000_000) + 978307200
    readable_date = datetime.datetime.fromtimestamp(unix_timestamp).isoformat()

    # Create Embedding
    content_to_embed = f"{name}: {text}"
    vector = model.encode(f"search_query: {content_to_embed}")

    data.append({
        "vector": vector.tolist(),
        "text": text,
        "timestamp": readable_date,
        "sender_name": name,
        "message_id": message_id
    })

    # Show first 5 messages for verification
    if i < 5:
        print(f"\nMessage {i + 1} sample:")
        print(f"  Handle: {handle_id}")
        print(f"  Resolved to: {name}")
        print(f"  Text: {text[:80]}..." if len(text) > 80 else f"  Text: {text}")

    # Progress indicator
    if (i + 1) % batch_size == 0:
        print(f"Processed {i + 1}/{len(rows)} messages")

conn.close()

# --- 5. Create LanceDB Table with Schema ---
schema = pa.schema([
    pa.field("vector", pa.list_(pa.float32(), 768)),
    pa.field("text", pa.string()),
    pa.field("timestamp", pa.string()),
    pa.field("sender_name", pa.string()),
    pa.field("message_id", pa.int64())
])

# Drop existing table if it exists and create new one
try:
    db.drop_table("imessages")
except Exception:
    pass  # Table doesn't exist, that's fine

table = db.create_table("imessages", data=data, schema=schema)

print(f"\nSuccessfully stored {len(data)} embeddings in LanceDB")
print(f"Table 'imessages' created with {table.count_rows()} rows")
print(f"Database location: ./lancedb/")
