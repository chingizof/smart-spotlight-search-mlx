# smart-spotlight-search-mlx
Smart Spotlight Search is local, RAG-powered “Spotlight on steroids” for macOS. It indexes your files, runs semantic search over their contents, and lets you ask questions like “where’s my math homework PDF?” instead of remembering filenames — all on-device with MLX.



## Get started:
1. Access your messages history. Open system settings -> privacy -> disk access -> give access to terminal you use. Similarly, give Contacts access to your terminal
2. Copy your message history. Run  ```cp ~/Library/Messages/chat.db* ~/Desktop/smart-spotlight-search-mlx/chat-history```
3. To index messages, run ```python data-ingestion/imessages.py```
4. To run a search query, run ```python data-ingestion/search.py "dinner plans"```