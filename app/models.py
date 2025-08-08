from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Story:
    id: str
    filename: str
    character_primary: List[str]
    character_secondary: List[str]
    setting_primary: List[str]
    setting_secondary: List[str]
    theme_primary: List[str]
    theme_secondary: List[str]
    events_primary: List[str]
    events_secondary: List[str]
    emotions_primary: List[str]
    emotions_secondary: List[str]
    keywords: List[str]


@dataclass
class SearchResult:
    story: Story
    score: float
    matched_fields: Dict[str, float]


