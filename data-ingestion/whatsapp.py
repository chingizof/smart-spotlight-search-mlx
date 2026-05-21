"""WhatsApp Export Ingestion Pipeline

Parses WhatsApp chat export .txt files and indexes them into LanceDB,
reusing the same chunking and graph pipeline as iMessages.

Export instructions:
  iOS:     Open chat → tap name → Export Chat → Without Media
  Android: Open chat → ⋮ → More → Export Chat → Without Media

Usage:
    python data-ingestion/whatsapp.py "WhatsApp Chat with Alice.txt"
    python data-ingestion/whatsapp.py --dir exports/
    python data-ingestion/whatsapp.py chat.txt --my-name "John" --skip-graph
    python data-ingestion/whatsapp.py --reset --dir exports/
"""

import argparse
import datetime
import re
import sys
import zlib
from pathlib import Path
from typing import Optional

import lancedb
import pyarrow as pa
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))

from imessages import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    LANCEDB_PATH,
    Chunk,
    Message,
    build_graph_index,
    create_sliding_window_chunks,
    group_into_temporal_blocks,
    parse_date_filter,
)

WA_TABLE_NAME = "whatsapp_chunked"

# ── Line patterns ─────────────────────────────────────────────────────────────
# iOS:     [15/01/2024, 10:00:00] Alice: Hello
# iOS alt: [1/15/24, 10:00:00 AM] Alice: Hello
_IOS_RE = re.compile(
    r'^\[(\d{1,2}/\d{1,2}/\d{2,4}),\s*'
    r'(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?)\]\s*'
    r'([^:]+):\s*(.*)',
)
# Android: 15/01/2024, 10:00 - Alice: Hello
#          1/15/24, 10:00 AM - Alice: Hello
_ANDROID_RE = re.compile(
    r'^(\d{1,2}/\d{1,2}/\d{2,4}),\s*'
    r'(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?)\s+-\s+'
    r'([^:]+):\s*(.*)',
)

_DT_FORMATS = [
    "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
    "%d/%m/%y %H:%M:%S", "%d/%m/%y %H:%M",
    "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
    "%m/%d/%y %H:%M:%S", "%m/%d/%y %H:%M",
    "%d/%m/%Y %I:%M:%S %p", "%d/%m/%Y %I:%M %p",
    "%d/%m/%y %I:%M:%S %p", "%d/%m/%y %I:%M %p",
    "%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %I:%M %p",
    "%m/%d/%y %I:%M:%S %p", "%m/%d/%y %I:%M %p",
]

_SYSTEM_RE = re.compile(
    r"Messages and calls are end-to-end encrypted|"
    r"changed the subject|changed this group|added .+|"
    r"\bleft\b|removed .+|created group|changed the group icon|"
    r"joined using this group|was added|"
    r"This message was deleted|You deleted this message|"
    r"security code changed|tap to learn more",
    re.IGNORECASE,
)

_MEDIA_RE = re.compile(
    r"^(<Media omitted>|image omitted|video omitted|audio omitted|"
    r"document omitted|GIF omitted|sticker omitted|Contact card omitted|"
    r"Poll created|Voice message omitted)\s*$",
    re.IGNORECASE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_timestamp(date_str: str, time_str: str) -> Optional[float]:
    combined = f"{date_str.strip()} {time_str.strip()}"
    for fmt in _DT_FORMATS:
        try:
            return datetime.datetime.strptime(combined, fmt).timestamp()
        except ValueError:
            continue
    return None


def _chat_id_from_path(path: Path) -> int:
    """Stable int ID from filename, offset above typical iMessage chat_id range."""
    return 1_000_000_000 + (zlib.crc32(path.name.encode()) & 0x3FFFFFFF)


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_export(path: Path, my_name: Optional[str] = None) -> list[Message]:
    """
    Parse a WhatsApp .txt export into a list of Message objects.
    Multi-line messages are stitched together. System messages and
    media placeholders are dropped.
    """
    chat_id = _chat_id_from_path(path)
    messages: list[Message] = []
    msg_id = 0

    current_ts: Optional[float] = None
    current_sender: Optional[str] = None
    current_lines: list[str] = []

    def flush():
        nonlocal msg_id
        if current_ts is None or not current_lines:
            return
        text = " ".join(current_lines).strip()
        if not text or _SYSTEM_RE.search(text) or _MEDIA_RE.match(text):
            return
        is_from_me = (
            my_name is not None
            and current_sender is not None
            and current_sender.strip().lower() == my_name.strip().lower()
        )
        messages.append(Message(
            rowid=msg_id,
            chat_id=chat_id,
            handle=current_sender or "Unknown",
            text=text,
            timestamp=current_ts,
            is_from_me=is_from_me,
        ))
        msg_id += 1

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        m = _IOS_RE.match(line) or _ANDROID_RE.match(line)
        if m:
            flush()
            date_s, time_s, sender, text = m.group(1), m.group(2), m.group(3), m.group(4)
            ts = _parse_timestamp(date_s, time_s)
            if ts is None:
                # Unparseable timestamp — treat as continuation
                if current_ts is not None and line:
                    current_lines.append(line)
                continue
            current_ts = ts
            current_sender = sender.strip()
            current_lines = [text] if text.strip() else []
        elif current_ts is not None and line:
            current_lines.append(line)

    flush()
    return messages


# ── Storage ───────────────────────────────────────────────────────────────────

def _schema() -> pa.Schema:
    return pa.schema([
        pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
        pa.field("raw_text", pa.string()),
        pa.field("chat_id", pa.int64()),
        pa.field("start_timestamp", pa.string()),
        pa.field("end_timestamp", pa.string()),
        pa.field("message_ids", pa.list_(pa.int64())),
        pa.field("source", pa.string()),
    ])


def _last_indexed_ts(db: lancedb.DBConnection, chat_id: int) -> float:
    """Return the latest end_timestamp already in the table for this chat_id."""
    if WA_TABLE_NAME not in db.table_names():
        return 0.0
    df = db.open_table(WA_TABLE_NAME).to_pandas()
    rows = df[df["chat_id"] == chat_id]
    if rows.empty:
        return 0.0
    return max(
        datetime.datetime.fromisoformat(ts).timestamp()
        for ts in rows["end_timestamp"]
    )


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_export(
    path: Path,
    db: lancedb.DBConnection,
    model: SentenceTransformer,
    my_name: Optional[str] = None,
    skip_graph: bool = False,
) -> int:
    """Ingest one WhatsApp export. Returns number of chunks indexed."""
    print(f"\n  Processing: {path.name}")

    chat_id = _chat_id_from_path(path)
    source_label = path.stem

    messages = parse_export(path, my_name=my_name)
    if not messages:
        print("  ⚠ No messages parsed (check format)")
        return 0
    print(f"  Parsed {len(messages)} messages")

    last_ts = _last_indexed_ts(db, chat_id)
    if last_ts > 0:
        before = len(messages)
        messages = [m for m in messages if m.timestamp > last_ts]
        print(f"  Incremental: {before - len(messages)} already indexed, {len(messages)} new")

    if not messages:
        print("  ✓ Already up to date")
        return 0

    blocks = group_into_temporal_blocks(messages)
    all_chunks: list[Chunk] = []
    for block in blocks:
        all_chunks.extend(create_sliding_window_chunks(block, chat_id, contact_map={}))

    if not all_chunks:
        print("  ⚠ No indexable chunks (messages too short)")
        return 0
    print(f"  {len(all_chunks)} chunks across {len(blocks)} temporal blocks")

    texts = [f"search_query: {c.embedding_text}" for c in all_chunks]
    vectors = model.encode(texts, batch_size=8, show_progress_bar=True, convert_to_numpy=True)

    rows = []
    for chunk, vec in zip(all_chunks, vectors):
        rows.append({
            "vector": vec.tolist(),
            "raw_text": chunk.raw_text,
            "chat_id": chunk.chat_id,
            "start_timestamp": datetime.datetime.fromtimestamp(chunk.start_timestamp).isoformat(),
            "end_timestamp": datetime.datetime.fromtimestamp(chunk.end_timestamp).isoformat(),
            "message_ids": chunk.message_ids,
            "source": source_label,
        })

    if WA_TABLE_NAME in db.table_names():
        db.open_table(WA_TABLE_NAME).add(rows)
    else:
        db.create_table(WA_TABLE_NAME, data=rows, schema=_schema())
    print(f"  ✓ Indexed {len(rows)} chunks into '{WA_TABLE_NAME}'")

    if not skip_graph:
        print("  Building graph index...")
        build_graph_index(all_chunks, model=model)

    return len(rows)


# ── Search ────────────────────────────────────────────────────────────────────

def search_whatsapp(
    query: str,
    limit: int = 5,
    model=None,
    after=None,
    before=None,
) -> list[dict]:
    """
    Vector search over WhatsApp chunks.
    Returns dicts with the same shape as imessages.search_memories,
    plus a 'source' key with the chat export filename stem.
    """
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)

    db = lancedb.connect(LANCEDB_PATH)
    if WA_TABLE_NAME not in db.table_names():
        return []

    table = db.open_table(WA_TABLE_NAME)
    query_vec = model.encode(f"search_query: {query}")
    search = table.search(query_vec).metric("cosine")

    after_dt = parse_date_filter(after)
    before_dt = parse_date_filter(before)
    if after_dt:
        search = search.where(f"start_timestamp >= '{after_dt.isoformat()}'")
    if before_dt:
        search = search.where(f"start_timestamp <= '{before_dt.isoformat()}'")

    results = search.limit(limit).to_pandas()
    out = []
    for _, row in results.iterrows():
        start_dt = datetime.datetime.fromisoformat(row["start_timestamp"])
        end_dt = datetime.datetime.fromisoformat(row["end_timestamp"])
        out.append({
            "text": row["raw_text"],
            "start_time": start_dt.strftime("%Y-%m-%d %H:%M"),
            "end_time": end_dt.strftime("%H:%M"),
            "message_count": len(row["message_ids"]),
            "distance": row["_distance"],
            "source": row.get("source", "WhatsApp"),
        })
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Index WhatsApp chat export .txt files into LanceDB"
    )
    parser.add_argument("files", nargs="*", help="Export .txt file(s) to index")
    parser.add_argument("--dir", type=Path, help="Directory of .txt exports to process")
    parser.add_argument(
        "--my-name",
        type=str,
        help="Your display name in the export (marks your messages as 'Me')",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and reindex the entire WhatsApp table",
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Skip knowledge graph indexing (faster, vector search only)",
    )
    parser.add_argument("--search", type=str, help="Test search after indexing")
    args = parser.parse_args()

    paths: list[Path] = []
    if args.dir:
        paths.extend(sorted(args.dir.glob("*.txt")))
    for f in args.files:
        paths.append(Path(f))

    if not paths:
        parser.error("Provide at least one .txt file or --dir <directory>")

    print("=" * 60)
    print("WhatsApp Export Ingestion Pipeline")
    print("=" * 60)

    print(f"\n  Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    db = lancedb.connect(LANCEDB_PATH)

    if args.reset and WA_TABLE_NAME in db.table_names():
        db.drop_table(WA_TABLE_NAME)
        print(f"  Reset: dropped '{WA_TABLE_NAME}'")

    total = 0
    for path in paths:
        if not path.exists():
            print(f"\n  ⚠ File not found: {path}")
            continue
        total += ingest_export(
            path, db, model,
            my_name=args.my_name,
            skip_graph=args.skip_graph,
        )

    print(f"\n✅ Done — {total} chunks indexed across {len(paths)} file(s)")

    if args.search:
        print(f"\n  Searching: '{args.search}'")
        results = search_whatsapp(args.search, model=model)
        if not results:
            print("  No results found.")
        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} [{r['source']}] dist={r['distance']:.4f} ---")
            print(f"[{r['start_time']} – {r['end_time']}] ({r['message_count']} msgs)")
            print(r["text"])


if __name__ == "__main__":
    main()
