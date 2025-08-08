## Book Search - Docs

### Quickstart

1) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
# Optional but recommended locally (for local embeddings fallback)
pip install sentence-transformers
```

3) Run the server

```bash
# Use a port other than 5000 on macOS to avoid AirPlay conflicts
PORT=5050 python server.py
```

Open in browser: http://127.0.0.1:${PORT:-5050}

Health check:

```bash
curl http://127.0.0.1:${PORT:-5050}/health | jq .
```

### Testing

Run tests in the virtual environment. Tests default to the local embeddings provider and disable external services.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Notes:
- Unit tests validate the search engine keyword/semantic behavior and HTTP route validations.
- An end-to-end test boots the Flask app on a local port and verifies upload + search.
- Coverage output is enabled via `pytest.ini`.

### CSV data format

Expected headers:
`filename, character_primary, character_secondary, setting_primary, setting_secondary, theme_primary, theme_secondary, events_primary, events_secondary, emotions_primary, emotions_secondary, keywords`

Values can be Python-like list strings (e.g., `['Hero', 'Sidekick']`) or semicolon-separated (`Hero; Sidekick`).

Example row:
```csv
alpha.txt,Hero;,Sidekick;,City;,Alley;,Courage;,Friendship;,Battle;,Chase;,Hope;,Fear;,adventure; quest
```

### Environment variables

- `EMBEDDING_PROVIDER` one of `openai|google|cohere|huggingface|local`; if unset or missing API keys, the app falls back to a local model
- `PINECONE_API_KEY` enables Pinecone index creation/use
- `GOOGLE_API_KEY` enables Gemini-powered query translate/refine
- `SUPABASE_URL`, `SUPABASE_ANON_KEY` enable feedback persistence
- `SARVAM_API_KEY` enables voice search

When not set, endpoints still work with graceful degradation:
- Search uses local embeddings + keyword matching only
- Feedback may return empty or `503` where unavailable
- Voice search returns `503` if not configured

### Environment variables

- PINECONE_API_KEY: enable Pinecone semantic search index
- GOOGLE_API_KEY: enable Gemini-powered translate/refine for queries
- SUPABASE_URL, SUPABASE_ANON_KEY: enable feedback persistence
- SARVAM_API_KEY: enable voice search (speech-to-text)

When these are not set, the app gracefully degrades:

- Embeddings provider falls back to local `sentence-transformers`
- Feedback endpoints return empty data or 503 where appropriate
- Voice search returns 503
- Pinecone search is disabled and only keyword matching is applied

### Deployment notes

- The entrypoint is `server.py` (also used by Render via `render.yaml`).
- Production should use a WSGI server (gunicorn) instead of Flask dev server.

```bash
gunicorn -w 2 -b 0.0.0.0:${PORT:-5050} server:app
```

### Documentation index

- Architecture overview: `architecture.md`
- Endpoints reference: `endpoints.md`
- OpenAPI specification: `openapi.yaml`


