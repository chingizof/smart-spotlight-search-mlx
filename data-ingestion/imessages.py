"""
iMessage Ingestion Pipeline with Sliding Window Chunking

A production-ready local RAG ingestion system that reads iMessages from macOS chat.db
and indexes them into LanceDB using temporal grouping and sliding window chunking.

Features:
- Temporal grouping: Messages separated by >30min gap start new conversational blocks
- Sliding window: 7-message chunks with 3-message overlap
- Smart filtering: Skips noise messages for embedding, keeps them for context
- Incremental indexing: Only processes new messages since last run
- Graceful permission handling for macOS Full Disk Access requirements
"""

import argparse
import datetime
import lancedb
import re
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pyarrow as pa
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# === Configuration ===
SYSTEM_CHAT_DB = Path.home() / "Library" / "Messages" / "chat.db"
LOCAL_CHAT_DB = Path("chat-history") / "chat.db"
LANCEDB_PATH = "lancedb"
TABLE_NAME = "imessages_chunked"
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1"
EMBEDDING_DIM = 768

# Chunking parameters
TEMPORAL_GAP_MINUTES = 30  # Gap threshold to start new conversational block
WINDOW_SIZE = 7  # Messages per chunk
WINDOW_OVERLAP = 3  # Overlap between consecutive chunks
MIN_MEANINGFUL_CHARS = 20  # Minimum characters for a chunk to be indexed

# Noise patterns - messages to skip for embedding but keep for context
NOISE_PATTERNS = [
    r"^(ok|okay|k|kk)\.?$",
    r"^(lol|lmao|haha|hehe|😂|🤣)+$",
    r"^(yes|no|yeah|yep|nope|nah)\.?$",
    r"^(thanks|thx|ty)\.?$",
    r"^(hi|hey|hello|bye|goodbye)\.?$",
    r"^(👍|👎|❤️|😊|��|😮|🎉|👀)+$",
    r"^Loved ",  # iMessage reactions
    r"^Liked ",
    r"^Disliked ",
    r"^Laughed at ",
    r"^Emphasized ",
    r"^Questioned ",
]
NOISE_REGEX = re.compile("|".join(NOISE_PATTERNS), re.IGNORECASE)

# Relative time patterns for date parsing
RELATIVE_TIME_PATTERNS = {
    r"last\s*(\d+)\s*days?": lambda m: datetime.timedelta(days=int(m.group(1))),
    r"last\s*(\d+)\s*weeks?": lambda m: datetime.timedelta(weeks=int(m.group(1))),
    r"last\s*(\d+)\s*months?": lambda m: datetime.timedelta(days=int(m.group(1)) * 30),
    r"last\s*(\d+)\s*years?": lambda m: datetime.timedelta(days=int(m.group(1)) * 365),
    r"last\s*week": lambda m: datetime.timedelta(weeks=1),
    r"last\s*month": lambda m: datetime.timedelta(days=30),
    r"last\s*year": lambda m: datetime.timedelta(days=365),
    r"yesterday": lambda m: datetime.timedelta(days=1),
    r"today": lambda m: datetime.timedelta(days=0),
}


def parse_date_filter(date_str: Union[str, datetime.datetime, None]) -> Optional[datetime.datetime]:
    """
    Parse a date filter string into a datetime object.

    Supports:
    - ISO format dates: "2024-01-15", "2024-01-15T14:30:00"
    - Relative times: "last 7 days", "last week", "last month", "yesterday", "today"
    - datetime objects (passed through)
    - None (returns None)

    Returns:
        datetime object or None if input is None/empty
    """
    if date_str is None:
        return None

    if isinstance(date_str, datetime.datetime):
        return date_str

    if isinstance(date_str, datetime.date):
        return datetime.datetime.combine(date_str, datetime.time.min)

    date_str = date_str.strip().lower()
    if not date_str:
        return None

    # Try relative time patterns
    now = datetime.datetime.now()
    for pattern, delta_func in RELATIVE_TIME_PATTERNS.items():
        match = re.match(pattern, date_str, re.IGNORECASE)
        if match:
            delta = delta_func(match)
            # For "after" filters, we want the start of the period
            result = now - delta
            # Return start of day for consistency
            return result.replace(hour=0, minute=0, second=0, microsecond=0)

    # Try parsing as ISO format date/datetime
    for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d"]:
        try:
            return datetime.datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    raise ValueError(
        f"Cannot parse date '{date_str}'. Use ISO format (YYYY-MM-DD) or "
        f"relative format (e.g., 'last 7 days', 'last week', 'yesterday')"
    )


@dataclass
class Message:
    """Represents a single iMessage with metadata."""

    rowid: int
    chat_id: int  # Which conversation this message belongs to
    handle: Optional[str]
    text: str
    timestamp: float  # Unix timestamp
    is_from_me: bool

    @property
    def is_noise(self) -> bool:
        """Check if message is noise (reactions, short responses, etc.)."""
        return bool(NOISE_REGEX.match(self.text.strip()))

    @property
    def meaningful_length(self) -> int:
        """Get character count excluding whitespace."""
        return len(self.text.strip())


@dataclass
class Chunk:
    """A chunk of messages to be indexed."""

    messages: list[Message]
    chat_id: int
    start_timestamp: float
    end_timestamp: float
    contact_map: dict

    @property
    def message_ids(self) -> list[int]:
        return [m.rowid for m in self.messages]

    @property
    def raw_text(self) -> str:
        """Full text block for LLM context (includes noise)."""
        lines = []
        for msg in self.messages:
            sender = self._resolve_sender(msg)
            lines.append(f"[{sender}]: {msg.text}")
        return "\n".join(lines)

    @property
    def embedding_text(self) -> str:
        """Text for embedding (excludes noise)."""
        lines = []
        for msg in self.messages:
            if not msg.is_noise:
                sender = self._resolve_sender(msg)
                lines.append(f"{sender}: {msg.text}")
        return "\n".join(lines)

    @property
    def meaningful_char_count(self) -> int:
        """Count meaningful characters (excluding noise messages)."""
        return sum(msg.meaningful_length for msg in self.messages if not msg.is_noise)

    @property
    def is_indexable(self) -> bool:
        """Check if chunk has enough meaningful content to index."""
        return self.meaningful_char_count >= MIN_MEANINGFUL_CHARS

    def _resolve_sender(self, msg: Message) -> str:
        """Resolve message sender to contact name."""
        if msg.is_from_me:
            return "Me"
        if not msg.handle:
            return "Unknown"

        # Try phone number lookup (last 10 digits)
        if "@" not in msg.handle:
            key = re.sub(r"\D", "", msg.handle)[-10:]
        else:
            key = msg.handle.lower()

        return self.contact_map.get(key, msg.handle)


def copy_chat_database() -> tuple[bool, str]:
    """
    Copy the system chat.db to a local directory for safe read-only access.
    Returns (success, error_message).
    """
    # Create chat-history directory if it doesn't exist
    LOCAL_CHAT_DB.parent.mkdir(parents=True, exist_ok=True)

    # Check if system database exists
    if not SYSTEM_CHAT_DB.exists():
        return False, f"Messages database not found at {SYSTEM_CHAT_DB}"

    try:
        # Try to copy the database (this requires Full Disk Access)
        print(f"📋 Copying chat.db from {SYSTEM_CHAT_DB}...")
        shutil.copy2(SYSTEM_CHAT_DB, LOCAL_CHAT_DB)

        # Also copy the write-ahead log files if they exist (for consistency)
        for suffix in ["-wal", "-shm"]:
            wal_file = SYSTEM_CHAT_DB.with_suffix(SYSTEM_CHAT_DB.suffix + suffix)
            if wal_file.exists():
                shutil.copy2(wal_file, LOCAL_CHAT_DB.with_suffix(LOCAL_CHAT_DB.suffix + suffix))

        print(f"✓ Database copied to {LOCAL_CHAT_DB}")
        return True, ""

    except PermissionError:
        return False, (
            "Cannot access Messages database. Please grant Full Disk Access:\n"
            "  1. Open System Settings → Privacy & Security → Full Disk Access\n"
            "  2. Add your terminal app (Terminal, iTerm2, VS Code, etc.)\n"
            "  3. Restart your terminal and try again"
        )
    except Exception as e:
        return False, f"Failed to copy database: {e}"


def check_local_database() -> tuple[bool, str]:
    """
    Check if the local chat.db copy exists and is readable.
    Returns (can_access, error_message).
    """
    if not LOCAL_CHAT_DB.exists():
        return False, f"Local database not found at {LOCAL_CHAT_DB}. Run without --skip-copy first."

    try:
        conn = sqlite3.connect(f"file:{LOCAL_CHAT_DB}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM message LIMIT 1")
        conn.close()
        return True, ""
    except sqlite3.OperationalError as e:
        return False, f"Database error: {e}"


def load_contact_map() -> dict[str, str]:
    """Load contacts from macOS address book."""
    contact_map = {}

    try:
        import Contacts

        store = Contacts.CNContactStore.alloc().init()
        keys = [
            Contacts.CNContactGivenNameKey,
            Contacts.CNContactFamilyNameKey,
            Contacts.CNContactPhoneNumbersKey,
            Contacts.CNContactEmailAddressesKey,
        ]
        request = Contacts.CNContactFetchRequest.alloc().initWithKeysToFetch_(keys)

        def process_contact(contact, _stop):
            name = f"{contact.givenName()} {contact.familyName()}".strip() or "Unknown"
            for phone in contact.phoneNumbers():
                digits = re.sub(r"\D", "", phone.value().stringValue())
                if len(digits) >= 10:
                    contact_map[digits[-10:]] = name
            for email in contact.emailAddresses():
                contact_map[email.value().lower()] = name

        store.enumerateContactsWithFetchRequest_error_usingBlock_(request, None, process_contact)
        print(f"✓ Loaded {len(contact_map)} contacts from address book")

    except ImportError:
        print("⚠ Contacts framework not available - using raw handles")
    except Exception as e:
        print(f"⚠ Could not load contacts: {e}")

    return contact_map


def convert_imessage_timestamp(raw_date: int) -> float:
    """Convert iMessage timestamp (nanoseconds since 2001-01-01) to Unix timestamp."""
    return (raw_date / 1_000_000_000) + 978307200


def fetch_messages(conn: sqlite3.Connection, after_rowid: int = 0) -> list[Message]:
    """Fetch messages from chat.db with chat association, optionally after a specific ROWID."""
    query = """
    SELECT
        m.ROWID,
        cmj.chat_id,
        h.id AS contact_handle,
        m.text,
        m.date,
        m.is_from_me
    FROM message m
    JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
    LEFT JOIN handle h ON m.handle_id = h.ROWID
    WHERE m.text IS NOT NULL
      AND m.text != ''
      AND m.ROWID > ?
    ORDER BY cmj.chat_id ASC, m.date ASC
    """

    cur = conn.cursor()
    rows = cur.execute(query, (after_rowid,)).fetchall()

    messages = []
    for row in rows:
        rowid, chat_id, handle, text, raw_date, is_from_me = row
        messages.append(
            Message(
                rowid=rowid,
                chat_id=chat_id,
                handle=handle,
                text=text,
                timestamp=convert_imessage_timestamp(raw_date),
                is_from_me=bool(is_from_me),
            )
        )

    return messages


def group_by_chat(messages: list[Message]) -> dict[int, list[Message]]:
    """Group messages by chat_id, preserving chronological order within each chat."""
    chats: dict[int, list[Message]] = {}
    for msg in messages:
        if msg.chat_id not in chats:
            chats[msg.chat_id] = []
        chats[msg.chat_id].append(msg)
    return chats


def group_into_temporal_blocks(
    messages: list[Message], gap_minutes: int = TEMPORAL_GAP_MINUTES
) -> list[list[Message]]:
    """
    Group messages into conversational blocks based on time gaps.
    A new block starts when the gap between messages exceeds gap_minutes.
    Assumes all messages belong to the same chat.
    """
    if not messages:
        return []

    gap_seconds = gap_minutes * 60
    blocks = []
    current_block = [messages[0]]

    for msg in messages[1:]:
        time_gap = msg.timestamp - current_block[-1].timestamp

        if time_gap > gap_seconds:
            # Gap too large - start new block
            blocks.append(current_block)
            current_block = [msg]
        else:
            current_block.append(msg)

    # Don't forget the last block
    if current_block:
        blocks.append(current_block)

    return blocks


def create_sliding_window_chunks(
    block: list[Message],
    chat_id: int,
    contact_map: dict,
    window_size: int = WINDOW_SIZE,
    overlap: int = WINDOW_OVERLAP,
) -> list[Chunk]:
    """
    Create sliding window chunks from a temporal block.

    Args:
        block: List of messages in a conversational block (same chat)
        chat_id: The chat ID these messages belong to
        contact_map: Dictionary mapping handles to contact names
        window_size: Number of messages per chunk
        overlap: Number of messages to overlap between chunks
    """
    if not block:
        return []

    chunks = []
    step = window_size - overlap

    # Handle blocks smaller than window_size
    if len(block) < window_size:
        chunk = Chunk(
            messages=block,
            chat_id=chat_id,
            start_timestamp=block[0].timestamp,
            end_timestamp=block[-1].timestamp,
            contact_map=contact_map,
        )
        if chunk.is_indexable:
            chunks.append(chunk)
        return chunks

    # Sliding window
    i = 0
    while i < len(block):
        window_messages = block[i : i + window_size]

        # Create chunk
        chunk = Chunk(
            messages=window_messages,
            chat_id=chat_id,
            start_timestamp=window_messages[0].timestamp,
            end_timestamp=window_messages[-1].timestamp,
            contact_map=contact_map,
        )

        if chunk.is_indexable:
            chunks.append(chunk)

        # Move window
        i += step

        # Stop if we've covered all messages
        if i + window_size > len(block) and i < len(block):
            # Create final chunk with remaining messages
            final_messages = block[-window_size:]
            final_chunk = Chunk(
                messages=final_messages,
                chat_id=chat_id,
                start_timestamp=final_messages[0].timestamp,
                end_timestamp=final_messages[-1].timestamp,
                contact_map=contact_map,
            )
            # Only add if different from last chunk and indexable
            if chunks and final_chunk.message_ids != chunks[-1].message_ids:
                if final_chunk.is_indexable:
                    chunks.append(final_chunk)
            break

    return chunks


def get_last_indexed_rowid(db: lancedb.DBConnection, table_name: str) -> int:
    """Get the highest ROWID that has been indexed."""
    if table_name not in db.table_names():
        return 0

    table = db.open_table(table_name)
    df = table.to_pandas()

    if df.empty:
        return 0

    # message_ids is stored as a list, get max from all lists
    all_ids = []
    for ids in df["message_ids"]:
        if ids:
            all_ids.extend(ids)

    return max(all_ids) if all_ids else 0


def create_table_schema() -> pa.Schema:
    """Create PyArrow schema for LanceDB table."""
    return pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
            pa.field("raw_text", pa.string()),
            pa.field("chat_id", pa.int64()),
            pa.field("start_timestamp", pa.string()),
            pa.field("end_timestamp", pa.string()),
            pa.field("message_ids", pa.list_(pa.int64())),
        ]
    )


def search_memories(
    query: str,
    limit: int = 5,
    model=None,
    table=None,
    after: Union[str, datetime.datetime, None] = None,
    before: Union[str, datetime.datetime, None] = None,
) -> list[dict]:
    """
    Search for memories similar to the query with optional date filtering.

    Args:
        query: Search query text
        limit: Maximum number of results
        model: SentenceTransformer model (loads if not provided)
        table: LanceDB table (connects if not provided)
        after: Only return results after this date. Accepts:
            - ISO format string: "2024-01-15" or "2024-01-15T14:30:00"
            - Relative string: "last 7 days", "last week", "last month", "yesterday"
            - datetime object
        before: Only return results before this date. Same formats as 'after'.

    Returns:
        List of matching chunks with metadata
    """
    # Parse date filters
    after_dt = parse_date_filter(after)
    before_dt = parse_date_filter(before)

    # Load model if not provided
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)

    # Connect to table if not provided
    if table is None:
        db = lancedb.connect(LANCEDB_PATH)
        if TABLE_NAME not in db.table_names():
            print(f"Table '{TABLE_NAME}' not found. Run ingestion first.")
            return []
        table = db.open_table(TABLE_NAME)

    # Embed query
    query_vector = model.encode(f"search_query: {query}")

    # Build search with optional date filters
    search = table.search(query_vector).metric("cosine")

    # Apply date filters using LanceDB SQL-based filtering
    # start_timestamp is stored as ISO format string, which sorts lexicographically
    if after_dt is not None:
        after_iso = after_dt.isoformat()
        search = search.where(f"start_timestamp >= '{after_iso}'")

    if before_dt is not None:
        before_iso = before_dt.isoformat()
        search = search.where(f"start_timestamp <= '{before_iso}'")

    results = search.limit(limit).to_pandas()

    # Format results
    memories = []
    for _, row in results.iterrows():
        start_dt = datetime.datetime.fromisoformat(row["start_timestamp"])
        end_dt = datetime.datetime.fromisoformat(row["end_timestamp"])

        memories.append(
            {
                "text": row["raw_text"],
                "start_time": start_dt.strftime("%Y-%m-%d %H:%M"),
                "end_time": end_dt.strftime("%H:%M"),
                "message_count": len(row["message_ids"]),
                "distance": row["_distance"],
            }
        )

    return memories


def build_graph_index(chunks: list[Chunk], model) -> None:
    """
    Build or incrementally update the knowledge graph from a list of Chunk objects.
    Skips chunks already present in the graph (resumable after interruption).
    `model` is the already-loaded SentenceTransformer used for chunk embedding.
    """
    from graph_index import (
        add_chunk_to_graph,
        add_semantic_topic_edges,
        chunk_node_id,
        extract_speaker_names,
        load_graph,
        recompute_bm25_weights,
        save_graph,
    )
    from topic_extract import extract_topics

    G = load_graph()
    prev_node_by_chat: dict[int, str] = {}

    # Seed prev_node_by_chat from existing graph so temporal edges stay connected
    for node, attrs in G.nodes(data=True):
        if attrs.get("type") == "chunk":
            cid = attrs.get("chat_id", -1)
            cur = prev_node_by_chat.get(cid)
            if cur is None or node > cur:
                prev_node_by_chat[cid] = node

    new_count = 0
    all_new_topic_ids: list[str] = []

    for chunk in tqdm(chunks, desc="Building graph index", unit="chunk"):
        node_id = chunk_node_id(chunk.chat_id, chunk.start_timestamp)

        if G.has_node(node_id):
            prev_node_by_chat[chunk.chat_id] = node_id
            continue

        speakers = extract_speaker_names(chunk.raw_text)
        topics = extract_topics(chunk.raw_text)

        existing_people_lower = {p.lower() for p in topics["people"]}
        for name in speakers:
            if name.lower() not in existing_people_lower:
                topics["people"].append(name)

        start_time = datetime.datetime.fromtimestamp(chunk.start_timestamp).isoformat()
        end_time = datetime.datetime.fromtimestamp(chunk.end_timestamp).isoformat()

        new_topic_ids = add_chunk_to_graph(
            G=G,
            node_id=node_id,
            raw_text=chunk.raw_text,
            topics=topics,
            start_time=start_time,
            end_time=end_time,
            chat_id=chunk.chat_id,
            model=model,
            prev_node_id=prev_node_by_chat.get(chunk.chat_id),
        )

        all_new_topic_ids.extend(new_topic_ids)
        prev_node_by_chat[chunk.chat_id] = node_id
        new_count += 1

    if new_count > 0:
        print(f"  Adding semantic topic edges...")
        sem_edges = add_semantic_topic_edges(G, all_new_topic_ids)
        print(f"  Recomputing IDF weights...")
        recompute_bm25_weights(G)
        save_graph(G)
        print(
            f"✓ Graph updated: {new_count} chunks, "
            f"{len(all_new_topic_ids)} new topics, {sem_edges} semantic edges added"
        )
    else:
        print("✓ Graph index already up to date")


def build_graph_index_from_lancedb() -> None:
    """
    Retroactively build the graph from an existing LanceDB table.
    Useful when vector indexing was done before the graph was introduced.
    """
    from graph_index import (
        add_chunk_to_graph,
        add_semantic_topic_edges,
        chunk_node_id,
        extract_speaker_names,
        load_graph,
        recompute_bm25_weights,
        save_graph,
    )
    from topic_extract import extract_topics

    db = lancedb.connect(LANCEDB_PATH)
    if TABLE_NAME not in db.table_names():
        print(f"❌ Table '{TABLE_NAME}' not found. Run indexing first.")
        return

    table = db.open_table(TABLE_NAME)
    df = table.to_pandas()
    print(f"📊 Found {len(df)} chunks in LanceDB to process")

    print(f"🧠 Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)

    G = load_graph()
    prev_node_by_chat: dict[int, str] = {}
    new_count = 0
    all_new_topic_ids: list[str] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building graph index", unit="chunk"):
        start_dt = datetime.datetime.fromisoformat(row["start_timestamp"])
        node_id = chunk_node_id(int(row["chat_id"]), start_dt.timestamp())

        if G.has_node(node_id):
            prev_node_by_chat[int(row["chat_id"])] = node_id
            continue

        raw_text = row["raw_text"]
        speakers = extract_speaker_names(raw_text)
        topics = extract_topics(raw_text)

        existing_lower = {p.lower() for p in topics["people"]}
        for name in speakers:
            if name.lower() not in existing_lower:
                topics["people"].append(name)

        new_topic_ids = add_chunk_to_graph(
            G=G,
            node_id=node_id,
            raw_text=raw_text,
            topics=topics,
            start_time=row["start_timestamp"],
            end_time=row["end_timestamp"],
            chat_id=int(row["chat_id"]),
            model=model,
            prev_node_id=prev_node_by_chat.get(int(row["chat_id"])),
        )
        all_new_topic_ids.extend(new_topic_ids)
        prev_node_by_chat[int(row["chat_id"])] = node_id
        new_count += 1

    if new_count > 0:
        print(f"  Adding semantic topic edges...")
        sem_edges = add_semantic_topic_edges(G, all_new_topic_ids)
        print(f"  Recomputing IDF weights...")
        recompute_bm25_weights(G)
        save_graph(G)
        print(
            f"✓ Graph built: {new_count} chunks, "
            f"{len(all_new_topic_ids)} new topics, {sem_edges} semantic edges, "
            f"{G.number_of_nodes()} total nodes"
        )
    else:
        print("✓ Graph index already up to date")


def main():
    parser = argparse.ArgumentParser(
        description="Index iMessages into LanceDB with sliding window chunking"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Drop existing table and reindex all messages"
    )
    parser.add_argument("--search", type=str, help="Search for a memory after indexing")
    parser.add_argument("--search-only", type=str, help="Only search (skip indexing)")
    parser.add_argument(
        "--skip-copy",
        action="store_true",
        help="Skip copying chat.db (use existing local copy)",
    )
    parser.add_argument(
        "--after",
        type=str,
        help="Filter results after this date. Accepts ISO format (YYYY-MM-DD) or "
        "relative format (e.g., 'last 7 days', 'last week', 'yesterday')",
    )
    parser.add_argument(
        "--before",
        type=str,
        help="Filter results before this date. Accepts ISO format (YYYY-MM-DD) or "
        "relative format (e.g., 'last 7 days', 'last week', 'yesterday')",
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Skip knowledge graph indexing (faster, vector search only)",
    )
    parser.add_argument(
        "--build-graph-only",
        action="store_true",
        help="Build/update the knowledge graph from the existing LanceDB index (no message re-ingestion)",
    )
    args = parser.parse_args()

    # Handle build-graph-only mode
    if args.build_graph_only:
        print("\n🕸  Building knowledge graph from existing LanceDB index...")
        build_graph_index_from_lancedb()
        return

    # Handle search-only mode
    if args.search_only:
        filter_info = []
        if args.after:
            filter_info.append(f"after: {args.after}")
        if args.before:
            filter_info.append(f"before: {args.before}")
        filter_str = f" ({', '.join(filter_info)})" if filter_info else ""

        print(f"\n🔍 Searching for: '{args.search_only}'{filter_str}\n")
        try:
            results = search_memories(args.search_only, after=args.after, before=args.before)
        except ValueError as e:
            print(f"❌ {e}")
            return

        if not results:
            print("No results found.")
            return

        for i, mem in enumerate(results, 1):
            print(f"━━━ Result {i} (distance: {mem['distance']:.4f}) ━━━")
            print(f"📅 {mem['start_time']} - {mem['end_time']} ({mem['message_count']} messages)")
            print(f"{mem['text']}\n")
        return

    # === Check Database Access ===
    print("=" * 60)
    print("iMessage Ingestion Pipeline")
    print("=" * 60)

    # Copy or verify local database
    if args.skip_copy:
        print("\n📁 Using existing local database copy")
        can_access, error = check_local_database()
        if not can_access:
            print(f"\n❌ {error}")
            sys.exit(1)
    else:
        success, error = copy_chat_database()
        if not success:
            print(f"\n❌ {error}")
            sys.exit(1)

    print(f"✓ Database ready at {LOCAL_CHAT_DB}")

    # === Load Contacts ===
    contact_map = load_contact_map()

    # === Initialize LanceDB ===
    db = lancedb.connect(LANCEDB_PATH)

    if args.reset and TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)
        print(f"✓ Reset: dropped existing table '{TABLE_NAME}'")

    last_rowid = 0 if args.reset else get_last_indexed_rowid(db, TABLE_NAME)
    if last_rowid > 0:
        print(f"✓ Incremental mode: starting from ROWID {last_rowid}")

    # === Fetch Messages ===
    conn = sqlite3.connect(f"file:{LOCAL_CHAT_DB}?mode=ro", uri=True)
    messages = fetch_messages(conn, after_rowid=last_rowid)
    conn.close()

    print(f"\n📨 Found {len(messages)} new messages to process")

    if not messages:
        print("\n✅ Database is up to date - no new messages to index")
        if args.search:
            filter_info = []
            if args.after:
                filter_info.append(f"after: {args.after}")
            if args.before:
                filter_info.append(f"before: {args.before}")
            filter_str = f" ({', '.join(filter_info)})" if filter_info else ""

            print(f"\n🔍 Searching for: '{args.search}'{filter_str}\n")
            try:
                results = search_memories(args.search, after=args.after, before=args.before)
            except ValueError as e:
                print(f"❌ {e}")
                return

            for i, mem in enumerate(results, 1):
                print(f"━━━ Result {i} (distance: {mem['distance']:.4f}) ━━━")
                print(
                    f"📅 {mem['start_time']} - {mem['end_time']} ({mem['message_count']} messages)"
                )
                print(f"{mem['text']}\n")
        return

    # === Group by Chat, then by Time ===
    chats = group_by_chat(messages)
    print(f"📊 Found {len(chats)} distinct conversations")

    # Create temporal blocks within each chat
    all_blocks = []
    for chat_id, chat_messages in chats.items():
        blocks = group_into_temporal_blocks(chat_messages)
        for block in blocks:
            all_blocks.append((chat_id, block))

    print(f"📊 Grouped into {len(all_blocks)} conversational blocks across all chats")

    # Show block distribution
    block_sizes = [len(b) for _, b in all_blocks]
    print(
        f"   Block sizes: min={min(block_sizes)}, max={max(block_sizes)}, "
        f"avg={sum(block_sizes) / len(block_sizes):.1f}"
    )

    # === Create Sliding Window Chunks ===
    all_chunks = []
    for chat_id, block in all_blocks:
        chunks = create_sliding_window_chunks(block, chat_id, contact_map)
        all_chunks.extend(chunks)

    print(f"📝 Created {len(all_chunks)} indexable chunks")

    if not all_chunks:
        print("\n⚠ No chunks met the indexing criteria")
        return

    # Show sample chunk
    sample = all_chunks[0]
    print(f"\n📄 Sample chunk ({len(sample.messages)} messages):")
    print("-" * 40)
    for line in sample.raw_text.split("\n")[:3]:
        print(f"   {line[:70]}..." if len(line) > 70 else f"   {line}")
    if len(sample.messages) > 3:
        print(f"   ... and {len(sample.messages) - 3} more messages")
    print("-" * 40)

    # === Load Embedding Model ===
    print(f"\n🧠 Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)

    # === Generate Embeddings ===
    texts_to_embed = [f"search_query: {chunk.embedding_text}" for chunk in all_chunks]

    print(f"\n⚡ Encoding {len(texts_to_embed)} chunks...")
    start_time = time.time()

    vectors = model.encode(
        texts_to_embed, batch_size=8, show_progress_bar=True, convert_to_numpy=True
    )

    elapsed = time.time() - start_time
    print(
        f"✓ Encoding completed in {elapsed:.1f}s ({len(texts_to_embed) / elapsed:.1f} chunks/sec)"
    )

    # === Prepare Data for LanceDB ===
    data = []
    for i, chunk in enumerate(all_chunks):
        data.append(
            {
                "vector": vectors[i].tolist(),
                "raw_text": chunk.raw_text,
                "chat_id": chunk.chat_id,
                "start_timestamp": datetime.datetime.fromtimestamp(
                    chunk.start_timestamp
                ).isoformat(),
                "end_timestamp": datetime.datetime.fromtimestamp(chunk.end_timestamp).isoformat(),
                "message_ids": chunk.message_ids,
            }
        )

    # === Store in LanceDB ===
    schema = create_table_schema()

    if TABLE_NAME in db.table_names():
        table = db.open_table(TABLE_NAME)
        table.add(data)
        print(f"\n✓ Added {len(data)} chunks to existing table")
    else:
        table = db.create_table(TABLE_NAME, data=data, schema=schema)
        print(f"\n✓ Created new table with {len(data)} chunks")

    print(f"📊 Total chunks in database: {table.count_rows()}")
    print(f"📁 Database location: ./{LANCEDB_PATH}/")

    # === Knowledge Graph Indexing ===
    if not args.skip_graph:
        print("\n🕸  Building knowledge graph index...")
        build_graph_index(all_chunks, model=model)
    else:
        print("\n⏭  Skipping graph index (--skip-graph)")

    # === Optional Search Test ===
    if args.search:
        filter_info = []
        if args.after:
            filter_info.append(f"after: {args.after}")
        if args.before:
            filter_info.append(f"before: {args.before}")
        filter_str = f" ({', '.join(filter_info)})" if filter_info else ""

        print(f"\n🔍 Searching for: '{args.search}'{filter_str}\n")
        try:
            results = search_memories(
                args.search, model=model, table=table, after=args.after, before=args.before
            )
        except ValueError as e:
            print(f"❌ {e}")
            return

        for i, mem in enumerate(results, 1):
            print(f"━━━ Result {i} (distance: {mem['distance']:.4f}) ━━━")
            print(f"📅 {mem['start_time']} - {mem['end_time']} ({mem['message_count']} messages)")
            print(f"{mem['text']}\n")

    print("\n✅ Ingestion complete!")


if __name__ == "__main__":
    main()
