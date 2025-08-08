## API Endpoints

Base URL: `http://HOST:PORT`

### GET `/`
- Returns the HTML UI (`index.html`).

### GET `/health`
- Returns server status and embedding configuration.
- Response: `{ status, pinecone_connected, stories_loaded, embedding_provider, embedding_dimension }`

### POST `/search`
- Body: `{ "query": string, "top_k": number? }`
- Returns ranked results array with story metadata and matched field scores.

### POST `/upload`
- Multipart form upload with file field `file` containing a CSV.
- Loads/merges stories and (optionally) creates embeddings.

### POST `/submit-feedback`
- Body: `{ "query": string, "story_id": string, "feedback": string }`
- Persists a feedback record (requires Supabase env). Returns 503 if unavailable.

### POST `/voice-search`
- Multipart form upload with audio field `audio` (wav/mp3). Transcribes and searches.
- Returns 503 if voice service not configured.

### POST `/sync`
- Attempts to sync stories from Pinecone (no-op in refactor; keeps compatibility).

### POST `/delete-stories`
- Body: `{ "story_ids": string[] }`
- Deletes vectors (if Pinecone configured) and removes stories from local memory.

### GET `/list-stories`
- Lists in-memory stories with `id`, `filename`, `title`.

### GET `/get-feedback`
- Returns all feedback items if Supabase configured, else empty array.


