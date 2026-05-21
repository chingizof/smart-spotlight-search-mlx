"""Telegram Export Ingestion Pipeline

Parses Telegram Desktop chat export JSON files and indexes them into LanceDB,
reusing the same chunking and graph pipeline as iMessages and WhatsApp.

Export instructions:
  Telegram Desktop → open chat → hamburger menu → Export Chat History
  → Format: JSON → uncheck media → Export

The exported directory contains result.json (and ignored media subdirs).

Usage:
    python data-ingestion/telegram.py path/to/export/result.json
    python data-ingestion/telegram.py path/to/export-dir/     # finds result.json inside
    python data-ingestion/telegram.py --dir path/to/all-exports/
    python data-ingestion/telegram.py result.json --my-id 123456789
    python data-ingestion/telegram.py result.json --skip-graph --reset
"""

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Optional, Union

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

TG_TABLE_NAME = "telegram_chunked"

# Offset Telegram's native chat IDs above WhatsApp range (1B–2B) to avoid
# graph node collisions with iMessage (small) and WhatsApp (1B–2B).
_TG_CHAT_ID_OFFSET = 3_000_000_000


# ── Text extraction ───────────────────────────────────────────────────────────

def _extract_text(raw_text: Union[str, list]) -> str:
    """Flatten Telegram's text field, which can be a plain string or a list of
    entity dicts like {"type": "bold", "text": "hello"}."""
    if isinstance(raw_text, str):
        return raw_text
    if isinstance(raw_text, list):
        parts = []
        for entity in raw_text:
            if isinstance(entity, str):
                parts.append(entity)
            elif isinstance(entity, dict):
                parts.append(entity.get("text", ""))
        return "".join(parts)
    return ""


def _normalize_from_id(from_id: Union[str, int, None]) -> Optional[str]:
    """Return a canonical string form of from_id (strip 'user' prefix if present)."""
    if from_id is None:
        return None
    s = str(from_id)
    if s.startswith("user"):
        s = s[4:]
    return s


def _chat_id(tg_native_id: int) -> int:
    """Map Telegram's native chat ID into a range above iMessage / WhatsApp IDs."""
    return _TG_CHAT_ID_OFFSET + (abs(tg_native_id) % 1_000_000_000)


def _parse_ts(msg: dict) -> Optional[float]:
    """Extract Unix timestamp from a message dict."""
    raw = msg.get("date_unixtime")
    if raw is not None:
        try:
            return float(raw)
        except (ValueError, TypeError):
            pass
    date_str = msg.get("date")
    if date_str:
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.datetime.strptime(date_str, fmt).timestamp()
            except ValueError:
                continue
    return None


# ── Parsing ───────────────────────────────────────────────────────────────────

def _find_result_json(path: Path) -> Path:
    """Accept either a result.json file or the directory that contains it."""
    if path.is_file() and path.name == "result.json":
        return path
    if path.is_dir():
        candidate = path / "result.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"No result.json found in {path}")
    raise FileNotFoundError(f"Not a result.json or export directory: {path}")


def parse_export(
    path: Path,
    my_id: Optional[Union[int, str]] = None,
) -> tuple[list[Message], str, int]:
    """
    Parse a Telegram result.json export.

    Returns (messages, chat_name, native_chat_id).
    my_id should be your numeric Telegram user ID (marks your messages as is_from_me).
    For 'saved_messages' exports all messages are marked as yours automatically.
    """
    result_json = _find_result_json(path)
    data = json.loads(result_json.read_text(encoding="utf-8"))

    chat_name: str = data.get("name", "Telegram Chat")
    native_id: int = int(data.get("id", 0))
    chat_type: str = data.get("type", "")
    is_saved = (chat_type == "saved_messages")

    mapped_chat_id = _chat_id(native_id)
    my_id_str = _normalize_from_id(my_id) if my_id is not None else None

    messages: list[Message] = []

    for idx, msg in enumerate(data.get("messages", [])):
        if msg.get("type") != "message":
            continue

        ts = _parse_ts(msg)
        if ts is None:
            continue

        text = _extract_text(msg.get("text", "")).strip()
        if not text:
            continue

        sender = msg.get("from") or "Unknown"
        from_id_str = _normalize_from_id(msg.get("from_id"))

        if is_saved:
            is_from_me = True
        elif my_id_str is not None and from_id_str is not None:
            is_from_me = (from_id_str == my_id_str)
        else:
            is_from_me = False

        messages.append(Message(
            rowid=int(msg.get("id", idx)),
            chat_id=mapped_chat_id,
            handle=sender,
            text=text,
            timestamp=ts,
            is_from_me=is_from_me,
        ))

    return messages, chat_name, native_id


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
    if TG_TABLE_NAME not in db.table_names():
        return 0.0
    df = db.open_table(TG_TABLE_NAME).to_pandas()
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
    my_id: Optional[Union[int, str]] = None,
    skip_graph: bool = False,
) -> int:
    """Ingest one Telegram export directory or result.json. Returns chunks indexed."""
    try:
        result_json = _find_result_json(path)
    except FileNotFoundError as e:
        print(f"  ⚠ {e}")
        return 0

    print(f"\n  Processing: {result_json.parent.name or result_json.name}")

    messages, chat_name, native_id = parse_export(path, my_id=my_id)
    mapped_chat_id = _chat_id(native_id)

    if not messages:
        print("  ⚠ No messages parsed (check JSON format or file contents)")
        return 0
    print(f"  Chat: '{chat_name}'  |  {len(messages)} messages")

    last_ts = _last_indexed_ts(db, mapped_chat_id)
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
        all_chunks.extend(create_sliding_window_chunks(block, mapped_chat_id, contact_map={}))

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
            "source": chat_name,
        })

    if TG_TABLE_NAME in db.table_names():
        db.open_table(TG_TABLE_NAME).add(rows)
    else:
        db.create_table(TG_TABLE_NAME, data=rows, schema=_schema())
    print(f"  ✓ Indexed {len(rows)} chunks into '{TG_TABLE_NAME}'")

    if not skip_graph:
        print("  Building graph index...")
        build_graph_index(all_chunks, model=model)

    return len(rows)


# ── Search ────────────────────────────────────────────────────────────────────

def search_telegram(
    query: str,
    limit: int = 5,
    model=None,
    after=None,
    before=None,
) -> list[dict]:
    """
    Vector search over Telegram chunks.
    Returns dicts with the same shape as imessages.search_memories,
    plus a 'source' key with the Telegram chat name.
    """
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)

    db = lancedb.connect(LANCEDB_PATH)
    if TG_TABLE_NAME not in db.table_names():
        return []

    table = db.open_table(TG_TABLE_NAME)
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
            "source": row.get("source", "Telegram"),
        })
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def _collect_paths(files: list[str], dir_arg: Optional[Path]) -> list[Path]:
    """Gather result.json paths from explicit files/dirs and --dir."""
    paths: list[Path] = []
    if dir_arg:
        # Each subdirectory (or .json file) under --dir
        for entry in sorted(dir_arg.iterdir()):
            if entry.is_dir() and (entry / "result.json").exists():
                paths.append(entry)
            elif entry.is_file() and entry.name == "result.json":
                paths.append(entry)
    for f in files:
        paths.append(Path(f))
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Index Telegram chat export JSON files into LanceDB"
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="result.json file(s) or export director(ies) to index",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        help="Directory whose subdirectories each contain a result.json",
    )
    parser.add_argument(
        "--my-id",
        type=str,
        help="Your numeric Telegram user ID (marks your messages as 'Me'). "
             "Find it via @userinfobot on Telegram.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and reindex the entire Telegram table",
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Skip knowledge graph indexing (faster, vector search only)",
    )
    parser.add_argument("--search", type=str, help="Test search after indexing")
    args = parser.parse_args()

    paths = _collect_paths(args.files, args.dir)
    if not paths:
        parser.error("Provide at least one result.json / export directory or --dir")

    print("=" * 60)
    print("Telegram Export Ingestion Pipeline")
    print("=" * 60)

    print(f"\n  Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
    db = lancedb.connect(LANCEDB_PATH)

    if args.reset and TG_TABLE_NAME in db.table_names():
        db.drop_table(TG_TABLE_NAME)
        print(f"  Reset: dropped '{TG_TABLE_NAME}'")

    total = 0
    for path in paths:
        total += ingest_export(
            path, db, model,
            my_id=args.my_id,
            skip_graph=args.skip_graph,
        )

    print(f"\n✅ Done — {total} chunks indexed across {len(paths)} export(s)")

    if args.search:
        print(f"\n  Searching: '{args.search}'")
        results = search_telegram(args.search, model=model)
        if not results:
            print("  No results found.")
        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} [{r['source']}] dist={r['distance']:.4f} ---")
            print(f"[{r['start_time']} – {r['end_time']}] ({r['message_count']} msgs)")
            print(r["text"])


if __name__ == "__main__":
    main()
