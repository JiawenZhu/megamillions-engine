"""
utils.py — Shared helpers for Lottery Engine v6.0
==================================================
Now game-aware: all paths/column names come from the BaseGame instance.
Legacy API (no game arg, defaults to mega_millions CSV paths) is preserved.
"""

import pandas as pd
from collections import Counter

from games.base_game import BaseGame


def load_history(game: BaseGame | None = None, csv_path: str | None = None) -> pd.DataFrame:
    """
    Load and normalize the history CSV, sorted newest-first.
    Accepts either a BaseGame instance (preferred) or a legacy csv_path string.
    """
    if game is not None:
        path = game.csv_path
    elif csv_path is not None:
        path = csv_path
    else:
        path = "megamillions_history.csv"  # legacy default

    df = pd.read_csv(path)
    df["DrawDate"] = pd.to_datetime(df["DrawDate"], format="mixed", dayfirst=False)
    df = df.sort_values(by="DrawDate", ascending=False).reset_index(drop=True)
    return df


def calculate_frequency(
    df: pd.DataFrame,
    game: BaseGame | None = None,
) -> tuple[Counter, Counter]:
    """
    Return (wb_counts, sb_counts).
    If game is None, assumes legacy 5-ball Mega Millions schema.
    """
    if game:
        wb_cols = game.wb_cols
        sb_col  = game.sb_col
        sb_nums = game.all_sb_numbers()
    else:
        wb_cols = ["WB1", "WB2", "WB3", "WB4", "WB5"]
        sb_col  = "MegaBall"
        sb_nums = list(range(1, 26))

    white_balls = []
    special_balls = []

    for _, row in df.iterrows():
        for col in wb_cols:
            val = row.get(col)
            if pd.notna(val):
                white_balls.append(int(val))
        if sb_col:
            sb = row.get(sb_col)
            if pd.notna(sb):
                special_balls.append(int(sb))

    return Counter(white_balls), Counter(special_balls)


def calculate_overdue(
    df: pd.DataFrame,
    game: BaseGame | None = None,
) -> tuple[dict, dict]:
    """
    Return (wb_last_seen, sb_last_seen): days since each number was last drawn.
    Numbers never seen default to 999.
    """
    if game:
        wb_cols  = game.wb_cols
        sb_col   = game.sb_col
        all_wb   = game.all_wb_numbers()
        all_sb   = game.all_sb_numbers()
    else:
        wb_cols = ["WB1", "WB2", "WB3", "WB4", "WB5"]
        sb_col  = "MegaBall"
        all_wb  = list(range(1, 71))
        all_sb  = list(range(1, 26))

    wb_last_seen: dict[int, int] = {}
    sb_last_seen: dict[int, int] = {}

    # df is newest-first; first occurrence of a number = most recent draw
    for _, row in df.iterrows():
        days_ago = (pd.Timestamp.now() - row["DrawDate"]).days
        for col in wb_cols:
            val = row.get(col)
            if pd.notna(val):
                n = int(val)
                if n not in wb_last_seen:
                    wb_last_seen[n] = days_ago
        if sb_col:
            sb = row.get(sb_col)
            if pd.notna(sb):
                n = int(sb)
                if n not in sb_last_seen:
                    sb_last_seen[n] = days_ago

    # Fill unseen numbers
    for n in all_wb:
        wb_last_seen.setdefault(n, 999)
    for n in all_sb:
        sb_last_seen.setdefault(n, 999)

    return wb_last_seen, sb_last_seen
