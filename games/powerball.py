"""
games/powerball.py — Powerball game definition.

Draw days: Monday, Wednesday, Saturday  (Mon=0, Wed=2, Sat=5)
White balls: 5 from 1–69
Power Ball:  1 from 1–26
"""

from games.base_game import BaseGame


class Powerball(BaseGame):

    @property
    def name(self) -> str:
        return "Powerball"

    @property
    def slug(self) -> str:
        return "powerball"

    @property
    def draw_days(self) -> list[int]:
        return [0, 2, 5]  # Monday=0, Wednesday=2, Saturday=5

    # ── Ball Structure ────────────────────────────────────────────────────────

    @property
    def wb_range(self) -> tuple[int, int]:
        return (1, 69)

    @property
    def wb_count(self) -> int:
        return 5

    @property
    def sb_range(self) -> tuple[int, int]:
        return (1, 26)

    @property
    def sb_col(self) -> str:
        return "PowerBall"

    # ── Data Paths (in data/powerball/ subfolder) ────────────────────────────

    @property
    def csv_path(self) -> str:
        return "data/powerball/powerball_history.csv"

    @property
    def predictions_path(self) -> str:
        return "data/powerball/predictions.json"

    @property
    def mab_state_path(self) -> str:
        return "data/powerball/mab_state.json"

    @property
    def calibration_path(self) -> str:
        return "data/powerball/calibration.json"

    # ── Scraping ──────────────────────────────────────────────────────────────

    @property
    def scrape_url(self) -> str:
        return (
            "https://www.texaslottery.com/export/sites/lottery/Games/"
            "Powerball/Winning_Numbers/index.html"
        )

    # ── Payout Table ─────────────────────────────────────────────────────────

    @property
    def payout_table(self) -> dict[tuple, int]:
        return {
            (5, True):  20_000_000,   # Jackpot (minimum estimate)
            (5, False): 1_000_000,
            (4, True):  50_000,
            (4, False): 100,
            (3, True):  100,
            (3, False): 7,
            (2, True):  7,
            (2, False): 0,
            (1, True):  4,
            (1, False): 0,
            (0, True):  4,
            (0, False): 0,
        }
