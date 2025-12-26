import sqlite3
import Contacts
import re

# Load macOS contacts into a dictionary
contact_map = {}
store = Contacts.CNContactStore.alloc().init()
keys = [Contacts.CNContactGivenNameKey, Contacts.CNContactFamilyNameKey, 
        Contacts.CNContactPhoneNumbersKey, Contacts.CNContactEmailAddressesKey]
request = Contacts.CNContactFetchRequest.alloc().initWithKeysToFetch_(keys)

def process_contact(c, stop):
    name = f"{c.givenName()} {c.familyName()}".strip() or "Unknown"
    # Map phone numbers using last 10 digits to handle formatting diffs
    for p in c.phoneNumbers():
        digits = re.sub(r'\D', '', p.value().stringValue())
        if len(digits) >= 10:
            contact_map[digits[-10:]] = name
    # Map emails
    for e in c.emailAddresses():
        contact_map[e.value().lower()] = name

store.enumerateContactsWithFetchRequest_error_usingBlock_(request, None, process_contact)

# Query the database copy
conn = sqlite3.connect("chat-history/chat.db")
cur = conn.cursor()

q = """
SELECT 
    COALESCE(c.display_name, '') as group_name, 
    h.id as handle_id, 
    m.text
FROM message m
LEFT JOIN handle h ON m.handle_id = h.ROWID
LEFT JOIN chat_message_join j ON j.message_id = m.ROWID
LEFT JOIN chat c ON c.ROWID = j.chat_id
WHERE m.text IS NOT NULL
ORDER BY m.date ASC
"""

i = 0
for group_name, handle_id, text in cur.execute(q):
    # Determine name: Group Name > Contact Match > Raw Handle
    if group_name:
        display_name = group_name
    elif handle_id:
        key = re.sub(r'\D', '', handle_id)[-10:] if "@" not in handle_id else handle_id.lower()
        display_name = contact_map.get(key, handle_id)
    else:
        display_name = "Unknown"

    i += 1
    if i % 100 == 0:
        print(f"\n#{i} [{display_name}]")
        print(text)

conn.close()