# Smart Spotlight Search MLX

A local, privacy-first semantic search engine for macOS. Like Spotlight, but smarter — search your data using natural language queries instead of exact keywords. Everything runs on-device using Apple Silicon.

## Demo

![Demo](assets/demo.png)

## Features

- **Semantic Search**: Find messages by meaning, not just keywords ("dinner plans" finds "let's grab food tonight")
- **Conversational Context**: Messages are chunked into conversation blocks, preserving context for better retrieval
- **Date/Time Filtering**: Filter searches by date range ("last week", "2024-01-01 to 2024-01-31")
- **RAG Chat**: Ask questions about your messages using a local LLM (Llama 3.2)
- **Privacy-First**: All data stays on your device. No cloud, no external APIs
- **Apple Silicon Optimized**: Runs efficiently on M1/M2/M3 Macs
- **Incremental Indexing**: Only processes new messages on subsequent runs

## Currently Supported Data Sources

- [x] iMessages

## Quick Start

### 1. Grant Permissions

Open **System Settings → Privacy & Security** and grant your terminal:
- **Full Disk Access** (to read message history)
- **Contacts** (to resolve sender names)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Index Messages

```bash
python data-ingestion/imessages.py
```

This automatically copies your `chat.db` to a local `chat-history/` directory (for safety), then creates embeddings and stores them in LanceDB.

**Options:**
```bash
# Re-index everything from scratch
python data-ingestion/imessages.py --reset

# Skip copying chat.db (use existing local copy)
python data-ingestion/imessages.py --skip-copy

# Index and search immediately
python data-ingestion/imessages.py --search "dinner plans"

# Search only (skip indexing)
python data-ingestion/imessages.py --search-only "vacation photos"

# Search with date filters (works with --search or --search-only)
python data-ingestion/imessages.py --search-only "dinner" --after "last week"
python data-ingestion/imessages.py --search-only "meeting" --after "2024-01-01" --before "2024-01-31"
python data-ingestion/imessages.py --search-only "plans" --after "last 7 days"
```

### Date Filtering

Filter search results by date using the `--after` and `--before` flags:

**Relative formats:**
- `last 7 days`, `last 30 days`
- `last week`, `last month`, `last year`
- `yesterday`, `today`

**Absolute formats:**
- `2024-01-15` (ISO date)
- `2024-01-15T14:30:00` (ISO datetime)

### 4. Chat with RAG

Requires [Ollama](https://ollama.ai/) to be installed and running:

```bash
# Install Ollama
brew install ollama

# Start Ollama server (in a separate terminal)
ollama serve

# Run the chat interface
python data-ingestion/chat.py

# Run with date filters (only search messages from a specific time range)
python data-ingestion/chat.py --after "last week"
python data-ingestion/chat.py --after "2024-01-01" --before "2024-06-30"
```

The chat will automatically pull the `llama3.2` model on first run.

## How It Works

### Intelligent Chunking

Messages are processed using a sliding window approach that preserves conversational context:

1. **Chat Grouping**: Messages are first grouped by conversation (chat_id)
2. **Temporal Blocks**: Within each chat, messages are split into blocks when there's a >30 minute gap
3. **Sliding Window**: Each block is chunked into 7-message windows with 3-message overlap
4. **Smart Filtering**: Noise messages ("ok", "lol", reactions) are kept for context but excluded from embeddings

This means when you search, you get full conversation snippets instead of isolated messages.

### Example

When you search for "dinner plans", instead of getting:
```
[14:32] John: Sure, 7pm works
```

You get the full context:
```
[14:30 - 14:35]
[Me]: Hey, want to grab dinner tonight?
[John]: Sure! Where were you thinking?
[Me]: How about that new Thai place?
[John]: Sure, 7pm works
```

## Project Structure

```
smart-spotlight-search-mlx/
├── data-ingestion/
│   ├── imessages.py      # Index iMessages with sliding window chunking
│   └── chat.py           # RAG chat interface with Llama 3.2
├── tests/
│   ├── conftest.py       # Pytest fixtures with test database
│   └── test_imessages.py # Unit tests for ingestion pipeline
├── assets/               # Screenshots and demo images
├── chat-history/         # Local copy of chat.db (gitignored, auto-created)
├── lancedb/              # Vector database storage (gitignored)
└── requirements.txt
```

## Running Tests

The project includes a comprehensive test suite for the ingestion pipeline:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test class
pytest tests/test_imessages.py::TestParseDateFilter -v

# Run tests with coverage (if pytest-cov is installed)
pytest --cov=data-ingestion
```

### Test Coverage

The test suite covers:
- **Chunk size limits**: Ensures chunks never exceed 7 messages
- **Chat isolation**: Messages from different chats are never mixed
- **Temporal blocking**: >30 min gaps correctly split conversations
- **Noise filtering**: Reactions and short messages excluded from embeddings
- **Sliding window overlap**: Consecutive chunks share overlapping messages
- **Date filter parsing**: Relative ("last week") and absolute ("2024-01-15") formats

## Technical Details

- **Embedding Model**: [nomic-ai/nomic-embed-text-v1](https://huggingface.co/nomic-ai/nomic-embed-text-v1) (768 dimensions)
- **Vector Database**: [LanceDB](https://lancedb.com/) (local, serverless)
- **LLM**: [Llama 3.2 3B](https://ollama.com/library/llama3.2) via Ollama
- **Contact Resolution**: Uses macOS Contacts framework to map phone numbers/emails to names
- **Distance Metric**: Cosine similarity for semantic search

### Database Schema

| Field | Type | Description |
|-------|------|-------------|
| `vector` | float32[768] | Embedding of conversation chunk |
| `raw_text` | string | Full conversation text (for LLM context) |
| `chat_id` | int64 | Conversation identifier |
| `start_timestamp` | string | ISO timestamp of first message |
| `end_timestamp` | string | ISO timestamp of last message |
| `message_ids` | list[int64] | ROWIDs for incremental indexing |

### Chunking Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TEMPORAL_GAP_MINUTES` | 30 | Gap threshold to start new block |
| `WINDOW_SIZE` | 7 | Messages per chunk |
| `WINDOW_OVERLAP` | 3 | Overlap between consecutive chunks |
| `MIN_MEANINGFUL_CHARS` | 20 | Minimum chars to index a chunk |

## Roadmap

- [x] Add local LLM for RAG-powered Q&A
- [x] Sliding window chunking for better context
- [x] Per-conversation grouping
- [x] Add unit tests
- [ ] Index additional data sources:
  - [ ] Local files (PDFs, documents, notes)
  - [ ] Calendar events
  - [ ] Email (Apple Mail)
  - [ ] Safari browsing history
  - [ ] Notes app
- [ ] Build a native macOS UI (SwiftUI menubar app)
- [x] Add date/time filtering to search queries
- [ ] Hybrid search (combine semantic + keyword matching)
- [ ] Scheduled background indexing
- [ ] Add Whatsapp, Telegram messengers

## License

MIT
