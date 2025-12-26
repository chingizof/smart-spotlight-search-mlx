import sqlite3
import Contacts
import re
from sentence_transformers import SentenceTransformer

# Build Contact Map 
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

# Main Logic
DB = "chat-history/chat.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

q = """
SELECT
  m.ROWID,
  h.id AS contact_handle,
  m.text
FROM message m
LEFT JOIN handle h ON m.handle_id = h.ROWID
WHERE m.text IS NOT NULL
ORDER BY m.date ASC
"""

# Extract 1 message
row = cur.execute(q).fetchone()
handle_id = row[1]
text = row[2]

# Resolve Name
if handle_id:
    # Try to find name in contacts
    key = re.sub(r'\D', '', handle_id)[-10:] if "@" not in handle_id else handle_id.lower()
    name = contact_map.get(key, handle_id)
    
    # Check if name is still a raw phone number (+ and 11+ digits), mask it
    if re.match(r'^\+\d{11,}$', name):
        name = 'unknown'
else:
    name = "Me"

# Format: "Contact Name: text"
content_to_embed = f"{name}: {text}"
embedded_row = model.encode(f"search_query: {content_to_embed}")

print(f"Content: {content_to_embed}")
print(f"Embedding dimensions: {len(embedded_row)}")

conn.close()