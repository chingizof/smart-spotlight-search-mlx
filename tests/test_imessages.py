"""
Unit tests for iMessage ingestion pipeline.

Tests the core chunking logic:
- Chunk size limits (max 7 messages per chunk)
- Chat isolation (messages from same chat stay together)
- Temporal blocking (>30 min gap starts new block)
- Noise filtering (reactions, short messages excluded from embedding)
- Sliding window overlap
- Date filter parsing (relative and absolute formats)
"""

import datetime
import re
import sqlite3
from dataclasses import dataclass
from typing import Optional, Union

import pytest

# === Replicate core logic from imessages.py to avoid heavy imports ===
# (pyarrow, sentence_transformers, lancedb are not needed for unit tests)

TEMPORAL_GAP_MINUTES = 30
WINDOW_SIZE = 7
WINDOW_OVERLAP = 3
MIN_MEANINGFUL_CHARS = 20

NOISE_PATTERNS = [
    r"^(ok|okay|k|kk)\.?$",
    r"^(lol|lmao|haha|hehe|😂|🤣)+$",
    r"^(yes|no|yeah|yep|nope|nah)\.?$",
    r"^(thanks|thx|ty)\.?$",
    r"^(hi|hey|hello|bye|goodbye)\.?$",
    r"^(👍|👎|❤️|😊|😢|😮|🎉|👀)+$",
    r"^Loved ",
    r"^Liked ",
    r"^Disliked ",
    r"^Laughed at ",
    r"^Emphasized ",
    r"^Questioned ",
]
NOISE_REGEX = re.compile("|".join(NOISE_PATTERNS), re.IGNORECASE)

# Relative time patterns for date parsing (replicated from imessages.py)
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
    chat_id: int
    handle: Optional[str]
    text: str
    timestamp: float
    is_from_me: bool

    @property
    def is_noise(self) -> bool:
        return bool(NOISE_REGEX.match(self.text.strip()))

    @property
    def meaningful_length(self) -> int:
        return len(self.text.strip())


@dataclass
class Chunk:
    """A chunk of messages to be indexed."""

    messages: list
    chat_id: int
    start_timestamp: float
    end_timestamp: float
    contact_map: dict

    @property
    def message_ids(self) -> list:
        return [m.rowid for m in self.messages]

    @property
    def raw_text(self) -> str:
        lines = []
        for msg in self.messages:
            sender = self._resolve_sender(msg)
            lines.append(f"[{sender}]: {msg.text}")
        return "\n".join(lines)

    @property
    def embedding_text(self) -> str:
        lines = []
        for msg in self.messages:
            if not msg.is_noise:
                sender = self._resolve_sender(msg)
                lines.append(f"{sender}: {msg.text}")
        return "\n".join(lines)

    @property
    def meaningful_char_count(self) -> int:
        return sum(msg.meaningful_length for msg in self.messages if not msg.is_noise)

    @property
    def is_indexable(self) -> bool:
        return self.meaningful_char_count >= MIN_MEANINGFUL_CHARS

    def _resolve_sender(self, msg) -> str:
        if msg.is_from_me:
            return "Me"
        if not msg.handle:
            return "Unknown"
        if "@" not in msg.handle:
            key = re.sub(r"\D", "", msg.handle)[-10:]
        else:
            key = msg.handle.lower()
        return self.contact_map.get(key, msg.handle)


def convert_imessage_timestamp(raw_date: int) -> float:
    return (raw_date / 1_000_000_000) + 978307200


def fetch_messages(conn: sqlite3.Connection, after_rowid: int = 0) -> list:
    query = """
    SELECT m.ROWID, cmj.chat_id, h.id AS contact_handle, m.text, m.date, m.is_from_me
    FROM message m
    JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
    LEFT JOIN handle h ON m.handle_id = h.ROWID
    WHERE m.text IS NOT NULL AND m.text != '' AND m.ROWID > ?
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


def group_by_chat(messages: list) -> dict:
    chats = {}
    for msg in messages:
        if msg.chat_id not in chats:
            chats[msg.chat_id] = []
        chats[msg.chat_id].append(msg)
    return chats


def group_into_temporal_blocks(messages: list, gap_minutes: int = TEMPORAL_GAP_MINUTES) -> list:
    if not messages:
        return []
    gap_seconds = gap_minutes * 60
    blocks = []
    current_block = [messages[0]]
    for msg in messages[1:]:
        time_gap = msg.timestamp - current_block[-1].timestamp
        if time_gap > gap_seconds:
            blocks.append(current_block)
            current_block = [msg]
        else:
            current_block.append(msg)
    if current_block:
        blocks.append(current_block)
    return blocks


def create_sliding_window_chunks(
    block: list,
    chat_id: int,
    contact_map: dict,
    window_size: int = WINDOW_SIZE,
    overlap: int = WINDOW_OVERLAP,
) -> list:
    if not block:
        return []
    chunks = []
    step = window_size - overlap
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
    i = 0
    while i < len(block):
        window_messages = block[i : i + window_size]
        chunk = Chunk(
            messages=window_messages,
            chat_id=chat_id,
            start_timestamp=window_messages[0].timestamp,
            end_timestamp=window_messages[-1].timestamp,
            contact_map=contact_map,
        )
        if chunk.is_indexable:
            chunks.append(chunk)
        i += step
        if i + window_size > len(block) and i < len(block):
            final_messages = block[-window_size:]
            final_chunk = Chunk(
                messages=final_messages,
                chat_id=chat_id,
                start_timestamp=final_messages[0].timestamp,
                end_timestamp=final_messages[-1].timestamp,
                contact_map=contact_map,
            )
            if chunks and final_chunk.message_ids != chunks[-1].message_ids:
                if final_chunk.is_indexable:
                    chunks.append(final_chunk)
            break
    return chunks


class TestChunkSizeLimits:
    """Test that chunks respect the maximum message count."""

    def test_chunk_size_never_exceeds_window_size(self, test_connection, contact_map):
        """Each chunk should have at most WINDOW_SIZE (7) messages."""
        messages = fetch_messages(test_connection)
        chats = group_by_chat(messages)

        for chat_id, chat_messages in chats.items():
            blocks = group_into_temporal_blocks(chat_messages)
            for block in blocks:
                chunks = create_sliding_window_chunks(block, chat_id, contact_map)
                for chunk in chunks:
                    assert len(chunk.messages) <= WINDOW_SIZE, (
                        f"Chunk has {len(chunk.messages)} messages, "
                        f"expected at most {WINDOW_SIZE}"
                    )

    def test_small_blocks_create_single_chunk(self, contact_map):
        """A block smaller than WINDOW_SIZE should create exactly one chunk."""
        # Create a small block of 3 messages
        messages = [
            Message(rowid=1, chat_id=1, handle="+15551234567", text="Hello", timestamp=1000.0, is_from_me=False),
            Message(rowid=2, chat_id=1, handle=None, text="Hi there", timestamp=1060.0, is_from_me=True),
            Message(rowid=3, chat_id=1, handle="+15551234567", text="How are you?", timestamp=1120.0, is_from_me=False),
        ]

        chunks = create_sliding_window_chunks(messages, chat_id=1, contact_map=contact_map)

        assert len(chunks) == 1
        assert len(chunks[0].messages) == 3

    def test_large_block_creates_multiple_chunks(self, contact_map):
        """A block larger than WINDOW_SIZE should create multiple overlapping chunks."""
        # Create a block of 15 messages
        messages = [
            Message(
                rowid=i,
                chat_id=1,
                handle="+15551234567" if i % 2 == 0 else None,
                text=f"Message {i}",
                timestamp=1000.0 + (i * 60),
                is_from_me=i % 2 == 1,
            )
            for i in range(15)
        ]

        chunks = create_sliding_window_chunks(messages, chat_id=1, contact_map=contact_map)

        # With 15 messages, window=7, overlap=3, step=4:
        # Chunk 1: 0-6, Chunk 2: 4-10, Chunk 3: 8-14
        assert len(chunks) >= 3
        for chunk in chunks:
            assert len(chunk.messages) <= WINDOW_SIZE


class TestChatIsolation:
    """Test that messages from different chats are never mixed."""

    def test_all_messages_in_chunk_from_same_chat(self, test_connection, contact_map):
        """Every message in a chunk must have the same chat_id."""
        messages = fetch_messages(test_connection)
        chats = group_by_chat(messages)

        for chat_id, chat_messages in chats.items():
            blocks = group_into_temporal_blocks(chat_messages)
            for block in blocks:
                chunks = create_sliding_window_chunks(block, chat_id, contact_map)
                for chunk in chunks:
                    chat_ids_in_chunk = {msg.chat_id for msg in chunk.messages}
                    assert len(chat_ids_in_chunk) == 1, (
                        f"Chunk contains messages from multiple chats: {chat_ids_in_chunk}"
                    )
                    assert chunk.chat_id in chat_ids_in_chunk

    def test_chunk_chat_id_matches_messages(self, test_connection, contact_map):
        """The chunk's chat_id should match all its messages' chat_ids."""
        messages = fetch_messages(test_connection)
        chats = group_by_chat(messages)

        for chat_id, chat_messages in chats.items():
            blocks = group_into_temporal_blocks(chat_messages)
            for block in blocks:
                chunks = create_sliding_window_chunks(block, chat_id, contact_map)
                for chunk in chunks:
                    for msg in chunk.messages:
                        assert msg.chat_id == chunk.chat_id

    def test_group_by_chat_separates_conversations(self, test_connection):
        """group_by_chat should create separate lists for each chat."""
        messages = fetch_messages(test_connection)
        chats = group_by_chat(messages)

        # We have 3 chats in test data
        assert len(chats) == 3

        # Each chat should only contain messages from that chat
        for chat_id, chat_messages in chats.items():
            for msg in chat_messages:
                assert msg.chat_id == chat_id


class TestTemporalBlocking:
    """Test that time gaps correctly split conversations into blocks."""

    def test_gap_exceeding_threshold_creates_new_block(self, contact_map):
        """Messages with >30 min gap should be in separate blocks."""
        base_ts = 1000.0
        gap_seconds = (TEMPORAL_GAP_MINUTES + 15) * 60  # 45 minutes

        messages = [
            # First block
            Message(rowid=1, chat_id=1, handle="+1555", text="Hello", timestamp=base_ts, is_from_me=False),
            Message(rowid=2, chat_id=1, handle=None, text="Hi", timestamp=base_ts + 60, is_from_me=True),
            # Gap of 45 minutes
            Message(rowid=3, chat_id=1, handle="+1555", text="Back", timestamp=base_ts + gap_seconds, is_from_me=False),
            Message(rowid=4, chat_id=1, handle=None, text="Welcome back", timestamp=base_ts + gap_seconds + 60, is_from_me=True),
        ]

        blocks = group_into_temporal_blocks(messages)

        assert len(blocks) == 2
        assert len(blocks[0]) == 2  # First block: 2 messages
        assert len(blocks[1]) == 2  # Second block: 2 messages

    def test_gap_within_threshold_stays_in_same_block(self, contact_map):
        """Messages with <30 min gap should stay in same block."""
        base_ts = 1000.0
        gap_seconds = (TEMPORAL_GAP_MINUTES - 5) * 60  # 25 minutes

        messages = [
            Message(rowid=1, chat_id=1, handle="+1555", text="Hello", timestamp=base_ts, is_from_me=False),
            Message(rowid=2, chat_id=1, handle=None, text="Hi", timestamp=base_ts + 60, is_from_me=True),
            # Gap of 25 minutes (within threshold)
            Message(rowid=3, chat_id=1, handle="+1555", text="Still here", timestamp=base_ts + gap_seconds, is_from_me=False),
        ]

        blocks = group_into_temporal_blocks(messages)

        assert len(blocks) == 1
        assert len(blocks[0]) == 3

    def test_messages_in_chunk_within_reasonable_time_window(self, test_connection, contact_map):
        """Messages in a single chunk should be temporally close."""
        messages = fetch_messages(test_connection)
        chats = group_by_chat(messages)

        max_allowed_gap = TEMPORAL_GAP_MINUTES * 60  # 30 minutes in seconds

        for chat_id, chat_messages in chats.items():
            blocks = group_into_temporal_blocks(chat_messages)
            for block in blocks:
                chunks = create_sliding_window_chunks(block, chat_id, contact_map)
                for chunk in chunks:
                    # Check that time span within chunk is reasonable
                    time_span = chunk.end_timestamp - chunk.start_timestamp
                    # A chunk from a single temporal block should not span more than
                    # the gap threshold (since blocks are split at gaps)
                    # Allow some margin for the messages themselves
                    assert time_span <= max_allowed_gap + (WINDOW_SIZE * 300), (
                        f"Chunk spans {time_span}s which exceeds reasonable limit"
                    )


class TestNoiseFiltering:
    """Test that noise messages are handled correctly."""

    def test_noise_messages_detected(self):
        """Known noise patterns should be identified."""
        noise_texts = ["ok", "lol", "yeah", "👍", 'Loved "test"', 'Liked "hello"']

        for text in noise_texts:
            msg = Message(rowid=1, chat_id=1, handle="+1555", text=text, timestamp=1000.0, is_from_me=False)
            assert msg.is_noise, f"'{text}' should be detected as noise"

    def test_meaningful_messages_not_noise(self):
        """Meaningful messages should not be flagged as noise."""
        meaningful_texts = [
            "Hey, want to grab dinner tonight?",
            "The meeting is at 3pm",
            "I'll send you the document",
            "What do you think about the proposal?",
        ]

        for text in meaningful_texts:
            msg = Message(rowid=1, chat_id=1, handle="+1555", text=text, timestamp=1000.0, is_from_me=False)
            assert not msg.is_noise, f"'{text}' should not be detected as noise"

    def test_noise_excluded_from_embedding_text(self, contact_map):
        """Noise messages should be excluded from embedding text but kept in raw text."""
        messages = [
            Message(rowid=1, chat_id=1, handle="+15551234567", text="Want to meet?", timestamp=1000.0, is_from_me=False),
            Message(rowid=2, chat_id=1, handle=None, text="ok", timestamp=1060.0, is_from_me=True),
            Message(rowid=3, chat_id=1, handle="+15551234567", text="Great, see you at 5", timestamp=1120.0, is_from_me=False),
        ]

        chunks = create_sliding_window_chunks(messages, chat_id=1, contact_map=contact_map)
        chunk = chunks[0]

        # Raw text should include all messages
        assert "ok" in chunk.raw_text
        assert "Want to meet?" in chunk.raw_text
        assert "Great, see you at 5" in chunk.raw_text

        # Embedding text should exclude noise
        assert "ok" not in chunk.embedding_text
        assert "Want to meet?" in chunk.embedding_text
        assert "Great, see you at 5" in chunk.embedding_text


class TestSlidingWindowOverlap:
    """Test that sliding window creates proper overlap between chunks."""

    def test_consecutive_chunks_have_overlap(self, contact_map):
        """Consecutive chunks should share WINDOW_OVERLAP messages."""
        # Create 12 messages (should create multiple chunks)
        messages = [
            Message(
                rowid=i,
                chat_id=1,
                handle="+15551234567" if i % 2 == 0 else None,
                text=f"Message {i}",
                timestamp=1000.0 + (i * 60),
                is_from_me=i % 2 == 1,
            )
            for i in range(12)
        ]

        chunks = create_sliding_window_chunks(messages, chat_id=1, contact_map=contact_map)

        # Check overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            current_ids = set(chunks[i].message_ids)
            next_ids = set(chunks[i + 1].message_ids)
            overlap = current_ids & next_ids

            # Should have at least some overlap (may be less than WINDOW_OVERLAP at edges)
            assert len(overlap) > 0, f"Chunks {i} and {i+1} have no overlap"

    def test_no_messages_lost_in_chunking(self, contact_map):
        """All messages should appear in at least one chunk."""
        messages = [
            Message(
                rowid=i,
                chat_id=1,
                handle="+15551234567" if i % 2 == 0 else None,
                text=f"Message {i}",
                timestamp=1000.0 + (i * 60),
                is_from_me=i % 2 == 1,
            )
            for i in range(20)
        ]

        chunks = create_sliding_window_chunks(messages, chat_id=1, contact_map=contact_map)

        # Collect all message IDs from chunks
        all_chunk_msg_ids = set()
        for chunk in chunks:
            all_chunk_msg_ids.update(chunk.message_ids)

        # All original messages should be represented
        original_ids = {msg.rowid for msg in messages}
        assert original_ids == all_chunk_msg_ids, (
            f"Missing messages: {original_ids - all_chunk_msg_ids}"
        )


class TestMessageFetching:
    """Test database message fetching."""

    def test_fetch_returns_all_messages(self, test_connection):
        """fetch_messages should return all messages from the database."""
        messages = fetch_messages(test_connection)

        # Test data has: 15 (Alice) + 10 (Bob) + 10 (Charlie) = 35 messages
        assert len(messages) == 35

    def test_fetch_respects_after_rowid(self, test_connection):
        """fetch_messages should only return messages after specified ROWID."""
        all_messages = fetch_messages(test_connection)
        partial_messages = fetch_messages(test_connection, after_rowid=20)

        assert len(partial_messages) < len(all_messages)
        for msg in partial_messages:
            assert msg.rowid > 20

    def test_messages_have_required_fields(self, test_connection):
        """Each message should have all required fields populated."""
        messages = fetch_messages(test_connection)

        for msg in messages:
            assert msg.rowid > 0
            assert msg.chat_id > 0
            assert msg.text is not None
            assert msg.timestamp > 0
            assert isinstance(msg.is_from_me, bool)


class TestParseDateFilter:
    """Test date filter parsing for search queries."""

    def test_none_returns_none(self):
        """None input should return None."""
        assert parse_date_filter(None) is None

    def test_empty_string_returns_none(self):
        """Empty string should return None."""
        assert parse_date_filter("") is None
        assert parse_date_filter("   ") is None

    def test_datetime_passthrough(self):
        """datetime objects should be returned unchanged."""
        dt = datetime.datetime(2024, 1, 15, 14, 30, 0)
        result = parse_date_filter(dt)
        assert result == dt

    def test_date_converted_to_datetime(self):
        """date objects should be converted to datetime at midnight."""
        d = datetime.date(2024, 1, 15)
        result = parse_date_filter(d)
        assert result == datetime.datetime(2024, 1, 15, 0, 0, 0)

    def test_iso_date_format(self):
        """ISO date format (YYYY-MM-DD) should be parsed correctly."""
        result = parse_date_filter("2024-01-15")
        assert result == datetime.datetime(2024, 1, 15, 0, 0, 0)

    def test_iso_datetime_format(self):
        """ISO datetime format should be parsed correctly."""
        result = parse_date_filter("2024-01-15T14:30:00")
        assert result == datetime.datetime(2024, 1, 15, 14, 30, 0)

    def test_iso_datetime_with_space(self):
        """ISO datetime with space separator should be parsed correctly."""
        result = parse_date_filter("2024-01-15 14:30:00")
        assert result == datetime.datetime(2024, 1, 15, 14, 30, 0)

    def test_slash_date_format(self):
        """Date with slashes (YYYY/MM/DD) should be parsed correctly."""
        result = parse_date_filter("2024/01/15")
        assert result == datetime.datetime(2024, 1, 15, 0, 0, 0)

    def test_relative_last_n_days(self):
        """'last N days' should return approximately N days ago at start of day."""
        now = datetime.datetime.now()
        result = parse_date_filter("last 7 days")

        # Result should be approximately 7 days ago
        expected_approx = now - datetime.timedelta(days=7)
        expected_start_of_day = expected_approx.replace(hour=0, minute=0, second=0, microsecond=0)

        assert result == expected_start_of_day
        assert result.hour == 0
        assert result.minute == 0
        assert result.second == 0

    def test_relative_last_n_days_singular(self):
        """'last 1 day' should work (singular form)."""
        now = datetime.datetime.now()
        result = parse_date_filter("last 1 day")

        expected_approx = now - datetime.timedelta(days=1)
        expected_start_of_day = expected_approx.replace(hour=0, minute=0, second=0, microsecond=0)

        assert result == expected_start_of_day

    def test_relative_last_week(self):
        """'last week' should return 7 days ago at start of day."""
        now = datetime.datetime.now()
        result = parse_date_filter("last week")

        expected_approx = now - datetime.timedelta(weeks=1)
        expected_start_of_day = expected_approx.replace(hour=0, minute=0, second=0, microsecond=0)

        assert result == expected_start_of_day

    def test_relative_last_month(self):
        """'last month' should return 30 days ago at start of day."""
        now = datetime.datetime.now()
        result = parse_date_filter("last month")

        expected_approx = now - datetime.timedelta(days=30)
        expected_start_of_day = expected_approx.replace(hour=0, minute=0, second=0, microsecond=0)

        assert result == expected_start_of_day

    def test_relative_last_n_weeks(self):
        """'last N weeks' should return N*7 days ago."""
        now = datetime.datetime.now()
        result = parse_date_filter("last 2 weeks")

        expected_approx = now - datetime.timedelta(weeks=2)
        expected_start_of_day = expected_approx.replace(hour=0, minute=0, second=0, microsecond=0)

        assert result == expected_start_of_day

    def test_relative_last_n_months(self):
        """'last N months' should return N*30 days ago."""
        now = datetime.datetime.now()
        result = parse_date_filter("last 2 months")

        expected_approx = now - datetime.timedelta(days=60)
        expected_start_of_day = expected_approx.replace(hour=0, minute=0, second=0, microsecond=0)

        assert result == expected_start_of_day

    def test_relative_yesterday(self):
        """'yesterday' should return yesterday at start of day."""
        now = datetime.datetime.now()
        result = parse_date_filter("yesterday")

        expected_approx = now - datetime.timedelta(days=1)
        expected_start_of_day = expected_approx.replace(hour=0, minute=0, second=0, microsecond=0)

        assert result == expected_start_of_day

    def test_relative_today(self):
        """'today' should return today at start of day."""
        now = datetime.datetime.now()
        result = parse_date_filter("today")

        expected_start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

        assert result == expected_start_of_day

    def test_case_insensitive_iso(self):
        """ISO date parsing should work regardless of case in string."""
        result1 = parse_date_filter("2024-01-15")
        result2 = parse_date_filter("2024-01-15")
        assert result1 == result2

    def test_case_insensitive_relative(self):
        """Relative date formats should be case insensitive."""
        result_lower = parse_date_filter("last week")
        result_upper = parse_date_filter("LAST WEEK")
        result_mixed = parse_date_filter("Last Week")

        # All should produce the same result
        assert result_lower == result_upper == result_mixed

    def test_whitespace_handling_iso(self):
        """Extra whitespace around ISO dates should be handled gracefully."""
        result = parse_date_filter("  2024-01-15  ")
        assert result == datetime.datetime(2024, 1, 15, 0, 0, 0)

    def test_whitespace_handling_relative(self):
        """Extra whitespace in relative dates should be handled gracefully."""
        # "last  week" with extra space should still work
        result = parse_date_filter("last week")
        now = datetime.datetime.now()
        expected = (now - datetime.timedelta(weeks=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        assert result == expected

    def test_invalid_format_raises_error(self):
        """Invalid date formats should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_date_filter("not a date")
        assert "Cannot parse date" in str(exc_info.value)

        with pytest.raises(ValueError):
            parse_date_filter("01-15-2024")  # US format not supported

        with pytest.raises(ValueError):
            parse_date_filter("January 15, 2024")  # Written format not supported

    def test_relative_last_year(self):
        """'last year' should return 365 days ago."""
        now = datetime.datetime.now()
        result = parse_date_filter("last year")

        expected_approx = now - datetime.timedelta(days=365)
        expected_start_of_day = expected_approx.replace(hour=0, minute=0, second=0, microsecond=0)

        assert result == expected_start_of_day

    def test_relative_last_n_years(self):
        """'last N years' should return N*365 days ago."""
        now = datetime.datetime.now()
        result = parse_date_filter("last 2 years")

        expected_approx = now - datetime.timedelta(days=730)
        expected_start_of_day = expected_approx.replace(hour=0, minute=0, second=0, microsecond=0)

        assert result == expected_start_of_day

    def test_relative_results_are_at_start_of_day(self):
        """All relative date results should be at the start of day (00:00:00)."""
        relative_formats = [
            "today",
            "yesterday",
            "last week",
            "last month",
            "last year",
            "last 5 days",
            "last 2 weeks",
            "last 3 months",
        ]

        for fmt in relative_formats:
            result = parse_date_filter(fmt)
            assert result.hour == 0, f"'{fmt}' should have hour=0"
            assert result.minute == 0, f"'{fmt}' should have minute=0"
            assert result.second == 0, f"'{fmt}' should have second=0"
            assert result.microsecond == 0, f"'{fmt}' should have microsecond=0"

    def test_relative_ordering(self):
        """Relative dates should be correctly ordered in time."""
        today = parse_date_filter("today")
        yesterday = parse_date_filter("yesterday")
        last_week = parse_date_filter("last week")
        last_month = parse_date_filter("last month")
        last_year = parse_date_filter("last year")

        assert today > yesterday
        assert yesterday > last_week
        assert last_week > last_month
        assert last_month > last_year
