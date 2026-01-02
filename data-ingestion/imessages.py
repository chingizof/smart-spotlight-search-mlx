import sqlite3
import Contacts
import re
import datetime
import time
import sys
import lancedb
import pyarrow as pa
from sentence_transformers import SentenceTransformer

# Check for --reset flag
RESET = "--reset" in sys.argv

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

# Check for existing table and get already-indexed message IDs
existing_ids = set()
table_exists = "imessages" in db.list_tables()

if RESET and table_exists:
    db.drop_table("imessages")
    table_exists = False
    print("Reset flag detected - dropping existing table")

if table_exists:
    table = db.open_table("imessages")
    existing_ids = set(table.to_pandas()["message_id"].tolist())
    print(f"Found existing table with {len(existing_ids)} messages")

# --- 3. Load Model and Connect to SQLite ---
DB = "chat-history/chat.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()

# Load model (CPU is often faster than MPS for small embedding models on M2)
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

# --- 4. Process Only New Messages ---
rows = cur.execute(q).fetchall()
print(f"Found {len(rows)} total messages in chat.db")

# Filter out already-indexed messages
new_rows = [row for row in rows if row[0] not in existing_ids]
print(f"New messages to embed: {len(new_rows)}")

if len(new_rows) == 0:
    print("\nNo new messages to embed. Database is up to date!")
    conn.close()
    exit(0)

# First pass: prepare all metadata
metadata = []
texts_to_embed = []

for i, row in enumerate(new_rows):
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

    # Store metadata
    metadata.append({
        "text": text,
        "timestamp": readable_date,
        "sender_name": name,
        "message_id": message_id
    })

    # Prepare text for embedding
    content_to_embed = f"search_query: {name}: {text}"
    texts_to_embed.append(content_to_embed)

    # Show first 5 new messages for verification
    if i < 5:
        print(f"\nNew message {i + 1} sample:")
        print(f"  Handle: {handle_id}")
        print(f"  Resolved to: {name}")
        print(f"  Text: {text[:80]}..." if len(text) > 80 else f"  Text: {text}")

conn.close()

# Second pass: batch encode all texts
print(f"\nEncoding {len(texts_to_embed)} messages in batches...")
start_time = time.time()

batch_size = 32
vectors = model.encode(
    texts_to_embed,
    batch_size=batch_size,
    show_progress_bar=True,
    convert_to_numpy=True
)

elapsed = time.time() - start_time
print(f"Encoding completed in {elapsed:.1f}s ({len(texts_to_embed) / elapsed:.0f} messages/sec)")

# Combine vectors with metadata
data = []
for i, vec in enumerate(vectors):
    data.append({
        "vector": vec.tolist(),
        **metadata[i]
    })

# --- 5. Create or Update LanceDB Table ---
schema = pa.schema([
    pa.field("vector", pa.list_(pa.float32(), 768)),
    pa.field("text", pa.string()),
    pa.field("timestamp", pa.string()),
    pa.field("sender_name", pa.string()),
    pa.field("message_id", pa.int64())
])

if table_exists:
    # Append new data to existing table
    table.add(data)
    print(f"\nAdded {len(data)} new embeddings to existing table")
else:
    # Create new table
    table = db.create_table("imessages", data=data, schema=schema)
    print(f"\nCreated new table with {len(data)} embeddings")

print(f"Total messages in database: {table.count_rows()}")
print(f"Database location: ./lancedb/")
