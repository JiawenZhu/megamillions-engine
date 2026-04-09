"""
games/ — Lottery game configuration registry.

Usage:
    from games import get_game
    game = get_game("powerball")
    game = get_game("mega_millions")
    game = get_game("lotto")
"""

from games.mega_millions import MegaMillions
from games.powerball import Powerball
from games.lotto import Lotto

_REGISTRY = {
    "mega_millions": MegaMillions,
    "powerball":     Powerball,
    "lotto":         Lotto,
}


def get_game(name: str):
    """Return instantiated game config by key."""
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise ValueError(f"Unknown game '{name}'. Available: {available}")
    return _REGISTRY[key]()


def list_games() -> list[str]:
    return list(_REGISTRY.keys())
