import os
import sys
from pathlib import Path
import pytest

# Ensure repository root is importable during test collection
REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@pytest.fixture(autouse=True)
def _isolate_test_env(monkeypatch, tmp_path):
    # Ensure backup file does not pollute project directory
    monkeypatch.chdir(tmp_path)
    # Disable Pinecone and external services
    monkeypatch.setenv("PINECONE_API_KEY", "")
    monkeypatch.setenv("SUPABASE_URL", "")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "")
    monkeypatch.setenv("SARVAM_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("TEST_DB", "0")


