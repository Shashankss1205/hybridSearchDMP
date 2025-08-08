## Architecture Overview

The server is a modular Flask application with clear separation of concerns.

- `app/__init__.py`: application factory. Wires services, JSON encoder, CORS, and routes.
- `app/encoders.py`: custom `NumpyEncoder` for serializing NumPy types to JSON.
- `app/models.py`: dataclasses for `Story` and `SearchResult`.
- `app/routes.py`: all HTTP routes (JSON APIs and UI index page).
- `app/templates/index.html`: single-page UI (Tailwind CSS + vanilla JS) that talks to the API.
- `app/services/embeddings.py`: `APIEmbeddingModel` supporting OpenAI/Google/Cohere/HF APIs with local `sentence-transformers` fallback.
- `app/services/engine.py`: `StorySearchEngine` with hybrid scoring (semantic + keyword), Pinecone integration, CSV load, backup persistence.
- `app/services/feedback.py`: Supabase-backed feedback store.
- `app/services/voice.py`: SarvamAI Saarika-based speech-to-text handler.
- `server.py`: WSGI entrypoint, used both locally and in Render.

### Data flow

1. CSV upload (`/upload`) parses content into `Story` objects and persists a local backup.
2. If Pinecone is configured, embeddings are created and upserted per semantic field value.
3. Search (`/search`) cleans/refines query (if Google API set), runs semantic retrieval (Pinecone or local) and keyword fuzzy match, then merges and ranks.
4. UI renders results with per-field match indicators and enables feedback submission.

### Hybrid search

- Semantic: cosine similarity between query embedding and field embeddings (via Pinecone or local embeddings).
- Lexical: fuzzy keyword match for characters and keywords fields.
- Final score: combination of weights; semantic result aggregation uses max-over-fields and argmax per field.

### Degradation strategy

- Without `PINECONE_API_KEY`: semantic search limited to local embeddings.
- Without `GOOGLE_API_KEY`: query cleaning returns original query.
- Without `SUPABASE_*`: feedback endpoints return empty or 503.
- Without `SARVAM_API_KEY`: voice endpoint returns 503.


