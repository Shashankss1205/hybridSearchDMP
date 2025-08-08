import io
import os
import threading
import time

import requests

from server import app as wsgi_app  # type: ignore


def run_server_in_thread(port: int):
    wsgi_app.config.update({"TESTING": True})
    wsgi_app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


def test_e2e_flow(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("PINECONE_API_KEY", "")
    monkeypatch.setenv("SUPABASE_URL", "")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "")
    monkeypatch.setenv("SARVAM_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")

    port = 5678
    thread = threading.Thread(target=run_server_in_thread, args=(port,), daemon=True)
    thread.start()
    time.sleep(1.5)

    base = f"http://127.0.0.1:{port}"

    # Health
    r = requests.get(f"{base}/health", timeout=5)
    assert r.status_code == 200

    # Upload
    csv_content = (
        "filename,character_primary,character_secondary,setting_primary,setting_secondary,theme_primary,theme_secondary,events_primary,events_secondary,emotions_primary,emotions_secondary,keywords\n"
        "alpha.txt,Hero;,Sidekick;,City;,Alley;,Courage;,Friendship;,Battle;,Chase;,Hope;,Fear;,adventure; quest\n"
    ).encode("utf-8")
    files = {"file": ("stories.csv", io.BytesIO(csv_content), "text/csv")}
    r = requests.post(f"{base}/upload", files=files, timeout=10)
    assert r.status_code == 200

    # Search
    r = requests.post(f"{base}/search", json={"query": "hero adventure city"}, timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["total_found"] >= 1


