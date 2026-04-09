"""
games/mega_millions.py — Mega Millions game definition.

Draw days: Tuesday & Friday  (Tue=1, Fri=4)
White balls: 5 from 1–70
Mega Ball:   1 from 1–25
"""

from games.base_game import BaseGame


class MegaMillions(BaseGame):

    @property
    def name(self) -> str:
        return "Mega Millions"

    @property
    def slug(self) -> str:
        return "mega_millions"

    @property
    def draw_days(self) -> list[int]:
        return [1, 4]  # Tuesday=1, Friday=4

    # ── Ball Structure ────────────────────────────────────────────────────────

    @property
    def wb_range(self) -> tuple[int, int]:
        return (1, 70)

    @property
    def wb_count(self) -> int:
        return 5

    @property
    def sb_range(self) -> tuple[int, int]:
        return (1, 25)

    @property
    def sb_col(self) -> str:
        return "MegaBall"

    # ── Scraping ──────────────────────────────────────────────────────────────

    @property
    def scrape_url(self) -> str:
        return (
            "https://www.texaslottery.com/export/sites/lottery/Games/"
            "Mega_Millions/Winning_Numbers/index.html"
        )

    # ── Data Paths (in data/mega/ subfolder) ────────────────────────────────

    @property
    def csv_path(self) -> str:
        return "data/mega/megamillions_history.csv"

    @property
    def predictions_path(self) -> str:
        return "data/mega/predictions.json"

    @property
    def mab_state_path(self) -> str:
        return "data/mega/mab_state.json"

    @property
    def calibration_path(self) -> str:
        return "data/mega/calibration.json"

    # ── Payout Table ─────────────────────────────────────────────────────────

    @property
    def payout_table(self) -> dict[tuple, int]:
        return {
            (5, True):  20_000_000,   # Jackpot (minimum estimate)
            (5, False): 1_000_000,
            (4, True):  10_000,
            (4, False): 500,
            (3, True):  200,
            (3, False): 10,
            (2, True):  10,
            (2, False): 0,
            (1, True):  4,
            (1, False): 0,
            (0, True):  2,
            (0, False): 0,
        }
