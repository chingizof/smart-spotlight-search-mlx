"""
Pytest fixtures for iMessage ingestion tests.

Creates a temporary SQLite database with the same schema as macOS chat.db
and populates it with realistic test data.
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest


def create_test_database(db_path: Path) -> None:
    """Create a test database with the same schema as macOS chat.db."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create handle table (contacts)
    cur.execute("""
        CREATE TABLE handle (
            ROWID INTEGER PRIMARY KEY AUTOINCREMENT,
            id TEXT UNIQUE NOT NULL
        )
    """)

    # Create chat table (conversations)
    cur.execute("""
        CREATE TABLE chat (
            ROWID INTEGER PRIMARY KEY AUTOINCREMENT,
            guid TEXT UNIQUE NOT NULL,
            chat_identifier TEXT
        )
    """)

    # Create message table
    cur.execute("""
        CREATE TABLE message (
            ROWID INTEGER PRIMARY KEY AUTOINCREMENT,
            guid TEXT UNIQUE NOT NULL,
            text TEXT,
            handle_id INTEGER DEFAULT 0,
            date INTEGER,
            is_from_me INTEGER DEFAULT 0,
            FOREIGN KEY (handle_id) REFERENCES handle(ROWID)
        )
    """)

    # Create chat_message_join table (links messages to chats)
    cur.execute("""
        CREATE TABLE chat_message_join (
            chat_id INTEGER,
            message_id INTEGER,
            message_date INTEGER DEFAULT 0,
            PRIMARY KEY (chat_id, message_id),
            FOREIGN KEY (chat_id) REFERENCES chat(ROWID),
            FOREIGN KEY (message_id) REFERENCES message(ROWID)
        )
    """)

    conn.commit()
    conn.close()


def imessage_timestamp(unix_ts: float) -> int:
    """Convert Unix timestamp to iMessage timestamp (nanoseconds since 2001-01-01)."""
    return int((unix_ts - 978307200) * 1_000_000_000)


def populate_test_data(db_path: Path) -> None:
    """Populate the test database with realistic test data."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Insert handles (contacts)
    handles = [
        (1, "+15551234567"),  # Alice
        (2, "+15559876543"),  # Bob
        (3, "charlie@example.com"),  # Charlie
    ]
    cur.executemany("INSERT INTO handle (ROWID, id) VALUES (?, ?)", handles)

    # Insert chats (conversations)
    chats = [
        (1, "chat1", "+15551234567"),  # Chat with Alice
        (2, "chat2", "+15559876543"),  # Chat with Bob
        (3, "chat3", "charlie@example.com"),  # Chat with Charlie
    ]
    cur.executemany("INSERT INTO chat (ROWID, guid, chat_identifier) VALUES (?, ?, ?)", chats)

    # Base timestamp: 2024-01-15 10:00:00 UTC
    base_ts = 1705312800.0

    messages = []
    chat_message_joins = []
    msg_id = 1

    # === Chat 1 (Alice): 15 messages in quick succession (tests sliding window) ===
    for i in range(15):
        ts = base_ts + (i * 60)  # 1 minute apart
        is_from_me = i % 2  # Alternating
        text = f"Alice message {i + 1}" if not is_from_me else f"My reply {i + 1}"
        messages.append((msg_id, f"msg-alice-{i}", text, 1 if not is_from_me else None, imessage_timestamp(ts), is_from_me))
        chat_message_joins.append((1, msg_id, imessage_timestamp(ts)))
        msg_id += 1

    # === Chat 2 (Bob): Messages with a 45-minute gap (tests temporal blocking) ===
    # First block: 5 messages
    for i in range(5):
        ts = base_ts + (i * 120)  # 2 minutes apart
        is_from_me = i % 2
        text = f"Bob block1 msg {i + 1}" if not is_from_me else f"Reply block1 {i + 1}"
        messages.append((msg_id, f"msg-bob-b1-{i}", text, 2 if not is_from_me else None, imessage_timestamp(ts), is_from_me))
        chat_message_joins.append((2, msg_id, imessage_timestamp(ts)))
        msg_id += 1

    # Gap of 45 minutes (exceeds 30-minute threshold)
    gap_ts = base_ts + (4 * 120) + (45 * 60)

    # Second block: 5 messages
    for i in range(5):
        ts = gap_ts + (i * 120)
        is_from_me = i % 2
        text = f"Bob block2 msg {i + 1}" if not is_from_me else f"Reply block2 {i + 1}"
        messages.append((msg_id, f"msg-bob-b2-{i}", text, 2 if not is_from_me else None, imessage_timestamp(ts), is_from_me))
        chat_message_joins.append((2, msg_id, imessage_timestamp(ts)))
        msg_id += 1

    # === Chat 3 (Charlie): Short messages and noise (tests filtering) ===
    noise_messages = [
        "Hey Charlie!",
        "ok",
        "What's up?",
        "lol",
        'Loved "Hey Charlie!"',  # Reaction
        "Let's meet tomorrow",
        "yeah",
        "Sounds good!",
        "👍",
        "See you at 5pm",
    ]
    for i, text in enumerate(noise_messages):
        ts = base_ts + 3600 + (i * 60)  # Start 1 hour after base, 1 minute apart
        is_from_me = i % 2
        messages.append((msg_id, f"msg-charlie-{i}", text, 3 if not is_from_me else None, imessage_timestamp(ts), is_from_me))
        chat_message_joins.append((3, msg_id, imessage_timestamp(ts)))
        msg_id += 1

    # Insert all messages
    cur.executemany(
        "INSERT INTO message (ROWID, guid, text, handle_id, date, is_from_me) VALUES (?, ?, ?, ?, ?, ?)",
        messages,
    )
    cur.executemany(
        "INSERT INTO chat_message_join (chat_id, message_id, message_date) VALUES (?, ?, ?)",
        chat_message_joins,
    )

    conn.commit()
    conn.close()


@pytest.fixture
def test_db():
    """Create a temporary test database with realistic data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_chat.db"
        create_test_database(db_path)
        populate_test_data(db_path)
        yield db_path


@pytest.fixture
def test_connection(test_db):
    """Provide a database connection to the test database."""
    conn = sqlite3.connect(f"file:{test_db}?mode=ro", uri=True)
    yield conn
    conn.close()


@pytest.fixture
def contact_map():
    """Provide a test contact map."""
    return {
        "5551234567": "Alice",
        "5559876543": "Bob",
        "charlie@example.com": "Charlie",
    }
