"""
filters.py — v6.0 Data-Driven Rejection Filters
=================================================
Replaces the v5.0 hardcoded thresholds with dynamic values loaded from
the game's calibration JSON (produced by calibrate.py).

Usage:
    from filters import RejectionFilters
    f = RejectionFilters.from_game(game)
    if f.evaluate_all(combo):
        ...
"""

import json
import os


class RejectionFilters:
    """
    Statistically-grounded lottery ticket constraint checker.

    Parameters come from calibration JSON (data-driven), not guesswork.
    """

    def __init__(
        self,
        sum_low: float,
        sum_high: float,
        max_gap_limit: float,
        safe_even_min: int,
        safe_even_max: int,
        safe_high_min: int,
        safe_high_max: int,
    ):
        self.sum_low       = sum_low
        self.sum_high      = sum_high
        self.max_gap_limit = max_gap_limit
        self.safe_even_min = safe_even_min
        self.safe_even_max = safe_even_max
        self.safe_high_min = safe_high_min
        self.safe_high_max = safe_high_max

    @classmethod
    def from_calibration(cls, calibration_path: str) -> "RejectionFilters":
        """Load filter parameters from a pre-computed calibration JSON."""
        if not os.path.exists(calibration_path):
            raise FileNotFoundError(
                f"Calibration file not found: '{calibration_path}'. "
                "Run: python3.13 calibrate.py --game <name>"
            )
        with open(calibration_path) as f:
            cal = json.load(f)

        return cls(
            sum_low       = cal["sum"]["low_1_5sigma"],
            sum_high      = cal["sum"]["high_1_5sigma"],
            max_gap_limit = cal["gap"]["max_p99"],
            safe_even_min = cal["parity"]["safe_even_min"],
            safe_even_max = cal["parity"]["safe_even_max"],
            safe_high_min = cal["high_low"]["safe_high_min"],
            safe_high_max = cal["high_low"]["safe_high_max"],
        )

    @classmethod
    def from_game(cls, game) -> "RejectionFilters":
        """Convenience factory: load calibration from a BaseGame instance."""
        return cls.from_calibration(game.calibration_path)

    @classmethod
    def default_fallback(cls, game) -> "RejectionFilters":
        """
        Fallback if calibration file doesn't exist yet.
        Uses conservative math-based estimates (not from data).
        """
        lo, hi = game.wb_range
        mid = (lo + hi) / 2.0
        wb_n = game.wb_count
        # Theoretical mean of uniform draws
        theoretical_mean = wb_n * (lo + hi) / 2.0
        # Conservative std estimate
        theoretical_std = ((hi - lo) ** 2 / 12.0 * wb_n) ** 0.5
        return cls(
            sum_low       = theoretical_mean - 1.5 * theoretical_std,
            sum_high      = theoretical_mean + 1.5 * theoretical_std,
            max_gap_limit = (hi - lo) * 0.65,
            safe_even_min = 1,
            safe_even_max = wb_n - 1,
            safe_high_min = 1,
            safe_high_max = wb_n - 1,
        )

    # ── Individual filter methods ─────────────────────────────────────────────

    def sum_in_range(self, combo: list[int]) -> bool:
        """Reject tickets whose sum falls outside ±1.5σ of historical mean."""
        return self.sum_low <= sum(combo) <= self.sum_high

    def parity_balance(self, combo: list[int]) -> bool:
        """Reject all-odd or all-even tickets (and statistically rare extremes)."""
        evens = sum(1 for n in combo if n % 2 == 0)
        return self.safe_even_min <= evens <= self.safe_even_max

    def high_low_balance(self, combo: list[int], midpoint: float | None = None) -> bool:
        """Reject extreme clustering in high or low half of ball range."""
        if midpoint is None:
            # infer midpoint from combo range — used if called standalone
            midpoint = (min(combo) + max(combo)) / 2.0
        highs = sum(1 for n in combo if n > midpoint)
        return self.safe_high_min <= highs <= self.safe_high_max

    def consecutive_gap_limit(self, combo: list[int]) -> bool:
        """
        Two checks:
          1. Not fully consecutive (e.g., 1,2,3,4,5)
          2. No single gap exceeds the 99th-percentile historical maximum
        """
        sc   = sorted(combo)
        gaps = [sc[i+1] - sc[i] for i in range(len(sc)-1)]
        if all(g == 1 for g in gaps):
            return False
        if any(g > self.max_gap_limit for g in gaps):
            return False
        return True

    def evaluate_all(self, combo: list[int], midpoint: float | None = None) -> bool:
        """Run all filters. Returns True only if ticket passes every constraint."""
        return (
            self.sum_in_range(combo)
            and self.parity_balance(combo)
            and self.high_low_balance(combo, midpoint)
            and self.consecutive_gap_limit(combo)
        )

    def __repr__(self) -> str:
        return (
            f"<RejectionFilters sum=[{self.sum_low:.0f},{self.sum_high:.0f}] "
            f"gap<{self.max_gap_limit:.0f} "
            f"even=[{self.safe_even_min},{self.safe_even_max}] "
            f"high=[{self.safe_high_min},{self.safe_high_max}]>"
        )
