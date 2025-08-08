import io
import json

import pytest

from app import create_app


@pytest.fixture()
def client(monkeypatch):
    # Force local embeddings and disable external services
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("PINECONE_API_KEY", "")
    monkeypatch.setenv("SUPABASE_URL", "")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "")
    monkeypatch.setenv("SARVAM_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")

    app = create_app()
    app.config.update({"TESTING": True})
    return app.test_client()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "healthy"
    assert "embedding_provider" in data


def test_search_validation(client):
    resp = client.post("/search", json={})
    assert resp.status_code == 400


def test_upload_and_search_flow(client):
    # Minimal CSV with two rows and semantic/keyword fields
    csv_content = (
        "filename,character_primary,character_secondary,setting_primary,setting_secondary,theme_primary,theme_secondary,events_primary,events_secondary,emotions_primary,emotions_secondary,keywords\n"
        "alpha.txt,Hero;,Sidekick;,City;,Alley;,Courage;,Friendship;,Battle;,Chase;,Hope;,Fear;,adventure; quest\n"
        "beta.txt,Villain;,Henchman;,Dungeon;,Forest;,Greed;,Power;,Heist;,Escape;,Anger;,Guilt;,crime; mystery\n"
    ).encode("utf-8")

    data = {"file": (io.BytesIO(csv_content), "stories.csv")}
    resp = client.post("/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    up = resp.get_json()
    assert up["stories_loaded"] >= 2

    # Search by a keyword present in alpha row
    resp = client.post("/search", json={"query": "adventure hero in city", "top_k": 5})
    assert resp.status_code == 200
    result = resp.get_json()
    assert result["total_found"] >= 1
    assert any("alpha" in r["story"]["id"] for r in result["results"]) or any(
        "beta" in r["story"]["id"] for r in result["results"]
    )


