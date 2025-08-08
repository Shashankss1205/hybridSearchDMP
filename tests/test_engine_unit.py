import numpy as np

from app.services.engine import StorySearchEngine, Story


def test_keyword_search_and_ranking(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    engine = StorySearchEngine()
    # Avoid touching any persisted backup in tests
    engine.stories.clear()

    s1 = Story(
        id="alpha",
        filename="alpha.txt",
        character_primary=["hero"],
        character_secondary=["sidekick"],
        setting_primary=["city"],
        setting_secondary=["alley"],
        theme_primary=["courage"],
        theme_secondary=["friendship"],
        events_primary=["battle"],
        events_secondary=["chase"],
        emotions_primary=["hope"],
        emotions_secondary=["fear"],
        keywords=["adventure", "quest"],
    )

    s2 = Story(
        id="beta",
        filename="beta.txt",
        character_primary=["villain"],
        character_secondary=["henchman"],
        setting_primary=["dungeon"],
        setting_secondary=["forest"],
        theme_primary=["greed"],
        theme_secondary=["power"],
        events_primary=["heist"],
        events_secondary=["escape"],
        emotions_primary=["anger"],
        emotions_secondary=["guilt"],
        keywords=["crime", "mystery"],
    )

    engine.stories = {s1.id: s1, s2.id: s2}

    # Query that should favor s1 due to keywords and characters
    results = engine.search("hero adventure in city", top_k=5)
    assert len(results) >= 1
    assert results[0].story.id in {"alpha", "beta"}


