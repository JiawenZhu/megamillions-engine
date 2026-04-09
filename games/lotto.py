"""
games/lotto.py — Illinois Lotto game definition.

Draw days: Monday, Thursday, Saturday  (Mon=0, Thu=3, Sat=5)
White balls: 6 from 1–52
No special ball.

Payout reference:
  6/6 → Jackpot (starts $2M)
  5/6 → $1,500
  4/6 → $70
  3/6 → $3
  2/6 → Free ticket (we track as $2 equivalent)
"""

from games.base_game import BaseGame


class Lotto(BaseGame):

    @property
    def name(self) -> str:
        return "Illinois Lotto"

    @property
    def slug(self) -> str:
        return "lotto"

    @property
    def draw_days(self) -> list[int]:
        return [0, 3, 5]  # Monday=0, Thursday=3, Saturday=5

    # ── Ball Structure ────────────────────────────────────────────────────────

    @property
    def wb_range(self) -> tuple[int, int]:
        return (1, 52)

    @property
    def wb_count(self) -> int:
        return 6

    # sb_range and sb_col remain None (no special ball)

    # ── Data Paths (in data/lotto/ subfolder) ────────────────────────────────

    @property
    def csv_path(self) -> str:
        return "data/lotto/lotto_history.csv"

    @property
    def predictions_path(self) -> str:
        return "data/lotto/predictions.json"

    @property
    def mab_state_path(self) -> str:
        return "data/lotto/mab_state.json"

    @property
    def calibration_path(self) -> str:
        return "data/lotto/calibration.json"

    # ── Scraping ──────────────────────────────────────────────────────────────

    @property
    def scrape_url(self) -> str:
        return (
            "https://www.texaslottery.com/export/sites/lottery/Games/"
            "Lotto_Texas/Winning_Numbers/index.html"
        )

    # ── Payout Table ─────────────────────────────────────────────────────────

    @property
    def payout_table(self) -> dict[tuple, int]:
        # No special ball → sb_match is always False
        return {
            (6, False): 2_000_000,   # Jackpot minimum
            (5, False): 1_500,
            (4, False): 70,
            (3, False): 3,
            (2, False): 2,           # Free ticket tracked as face value
            (1, False): 0,
            (0, False): 0,
        }
