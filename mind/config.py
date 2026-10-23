import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    ollama_url: str = field(default_factory=lambda: os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    conscious_model: str = field(default_factory=lambda: os.environ.get("MIND_MODEL", "llama3.1:8b"))
    subconscious_model: str = field(default_factory=lambda: os.environ.get("MIND_SUB_MODEL", "llama3.2:3b"))
    db_path: str = field(default_factory=lambda: os.environ.get("MIND_DB", "mind.db"))
    short_term_turns: int = 12
    max_regenerations: int = 2
    max_delta_per_turn: float = 0.3
    # Seconds for an emotion to move halfway back to its baseline.
    decay_half_life: float = 1800.0
