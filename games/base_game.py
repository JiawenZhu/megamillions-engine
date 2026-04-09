"""
games/base_game.py — Abstract base class for all lottery game definitions.

Every game must define:
  - Identity (name, slug, draw days)
  - Ball structure (white ball range/count, special ball range or None)
  - Data paths (CSV, predictions, MAB state, calibration)
  - Payout table
  - Scrape URL
"""

from __future__ import annotations
from abc import ABC, abstractmethod


class BaseGame(ABC):
    """Abstract base class for a lottery game definition."""

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name, e.g. 'Mega Millions'."""
        ...

    @property
    @abstractmethod
    def slug(self) -> str:
        """File-safe identifier, e.g. 'mega_millions'."""
        ...

    @property
    @abstractmethod
    def draw_days(self) -> list[int]:
        """List of weekday indices (0=Mon … 6=Sun) for regular draws."""
        ...

    # ── Ball Structure ────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def wb_range(self) -> tuple[int, int]:
        """Inclusive (min, max) for white balls."""
        ...

    @property
    @abstractmethod
    def wb_count(self) -> int:
        """Number of white balls drawn."""
        ...

    @property
    def sb_range(self) -> tuple[int, int] | None:
        """Inclusive (min, max) for special ball; None if game has no special ball."""
        return None

    @property
    def sb_col(self) -> str | None:
        """Column name of the special ball in the history CSV; None if not applicable."""
        return None

    # ── Data Paths ────────────────────────────────────────────────────────────

    @property
    def csv_path(self) -> str:
        return f"data/{self.slug}_history.csv"

    @property
    def predictions_path(self) -> str:
        return f"data/{self.slug}_predictions.json"

    @property
    def mab_state_path(self) -> str:
        return f"data/{self.slug}_mab_state.json"

    @property
    def calibration_path(self) -> str:
        return f"data/{self.slug}_calibration.json"

    # ── Scraping ──────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def scrape_url(self) -> str:
        """Base URL for historical winning numbers."""
        ...

    @property
    def scrape_table_selector(self) -> str:
        """CSS selector for the results table rows."""
        return "table.large-only tbody tr"

    # ── Payout Table ─────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def payout_table(self) -> dict[tuple, int]:
        """
        Map (wb_matches: int, sb_match: bool) → prize $.
        For games with no special ball, use (wb_matches, False) always.
        """
        ...

    # ── Convenience ──────────────────────────────────────────────────────────

    @property
    def wb_cols(self) -> list[str]:
        """CSV column names for all white balls."""
        return [f"WB{i+1}" for i in range(self.wb_count)]

    def all_wb_numbers(self) -> list[int]:
        """All possible white ball numbers."""
        lo, hi = self.wb_range
        return list(range(lo, hi + 1))

    def all_sb_numbers(self) -> list[int]:
        """All possible special ball numbers, or empty list if no special ball."""
        if self.sb_range is None:
            return []
        lo, hi = self.sb_range
        return list(range(lo, hi + 1))

    def __repr__(self) -> str:
        return (
            f"<{self.name}: {self.wb_count} balls {self.wb_range}"
            + (f" + SB {self.sb_range}" if self.sb_range else "")
            + ">"
        )
