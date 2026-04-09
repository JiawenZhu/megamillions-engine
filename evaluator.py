"""
evaluator.py – Multi-Game Draw Evaluator v6.0
==============================================
Evaluates predictions for any supported lottery game.

Usage:
    python3.13 evaluator.py                       # default mega_millions
    python3.13 evaluator.py --game mega_millions
    python3.13 evaluator.py --game powerball
    python3.13 evaluator.py --game lotto
"""

import argparse
import json
import os

import pandas as pd

from games import get_game, list_games
from games.base_game import BaseGame
from mab_engine import (
    load_mab_state,
    save_mab_state,
    update_after_draw,
    print_mab_report,
)

TICKET_COST = 2  # dollars


def load_actual_results(game: BaseGame) -> dict:
    """Returns {date_str: {wb: set, mb: int|None}} for all draws in the game's CSV."""
    try:
        df = pd.read_csv(game.csv_path)
    except FileNotFoundError:
        print(f"Error: {game.csv_path} not found. Run scraper.py --game {game.slug} first.")
        return {}

    results = {}
    for _, row in df.iterrows():
        try:
            date_str = pd.to_datetime(
                row["DrawDate"], format="mixed", dayfirst=False
            ).strftime("%Y-%m-%d")

            wbs = set()
            for col in game.wb_cols:
                val = row.get(col)
                if pd.notna(val):
                    wbs.add(int(val))

            if len(wbs) != game.wb_count:
                continue

            mb = None
            if game.sb_col:
                mb_val = row.get(game.sb_col)
                if pd.notna(mb_val):
                    mb = int(mb_val)
                else:
                    continue  # special ball required but missing

            results[date_str] = {"wb": wbs, "mb": mb}
        except (ValueError, KeyError):
            continue
    return results


def evaluate_ticket(
    pred_wbs: list, pred_mb: int | None,
    actual_wbs: set, actual_mb: int | None,
    game: BaseGame,
) -> tuple[int, bool, int]:
    """Return (wb_matches, sb_match, prize)."""
    match_wbs = len(set(pred_wbs) & actual_wbs)
    match_sb  = (pred_mb == actual_mb) if (actual_mb is not None) else False
    prize     = game.payout_table.get((match_wbs, match_sb), 0)
    return match_wbs, match_sb, prize


def strategy_summary(predictions: list) -> dict:
    """Aggregate won/spent/roi per strategy label."""
    summary: dict[str, dict] = {}
    for p in predictions:
        s = p.get("strategy", "unknown")
        if s not in summary:
            summary[s] = {"tickets": 0, "spent": 0, "won": 0}
        summary[s]["tickets"] += 1
        summary[s]["spent"]   += TICKET_COST
        summary[s]["won"]     += p.get("won", 0)
    for s in summary:
        summary[s]["roi"] = summary[s]["won"] - summary[s]["spent"]
    return summary


def cumulative_report(all_runs: list, game: BaseGame) -> None:
    """Print a cross-run performance summary."""
    total_spent = 0
    total_won   = 0
    per_strategy: dict[str, dict] = {}

    for run in all_runs:
        if not run.get("evaluated"):
            continue
        for p in run["predictions"]:
            s   = p.get("strategy", "unknown")
            won = p.get("won", 0)
            if s not in per_strategy:
                per_strategy[s] = {"tickets": 0, "spent": 0, "won": 0}
            per_strategy[s]["tickets"] += 1
            per_strategy[s]["spent"]   += TICKET_COST
            per_strategy[s]["won"]     += won
            total_spent += TICKET_COST
            total_won   += won

    if not per_strategy:
        return

    print(f"\n{'='*55}")
    print(f"CUMULATIVE PERFORMANCE — {game.name.upper()}")
    print(f"  Total virtual spent : ${total_spent:,}")
    print(f"  Total virtual won   : ${total_won:,}")
    print(f"  Overall net ROI     : ${total_won - total_spent:,}")
    print("-" * 55)
    print(f"  {'Strategy':<12} {'Tickets':>8} {'Spent':>8} {'Won':>10} {'ROI':>10}")
    print("-" * 55)
    for s, d in sorted(per_strategy.items()):
        roi = d["won"] - d["spent"]
        print(
            f"  {s:<12} {d['tickets']:>8} ${d['spent']:>7,} "
            f"${d['won']:>9,} ${roi:>9,}"
        )
    print("=" * 55)


def main(game: BaseGame) -> None:
    print(f"\n🎰 Evaluator v6.0 — {game.name}")

    if not os.path.exists(game.predictions_path):
        print(f"No predictions found at {game.predictions_path}.")
        print(f"Run: python3.13 predictor.py --game {game.slug}")
        return

    actuals = load_actual_results(game)
    if not actuals:
        return

    with open(game.predictions_path, "r") as f:
        all_runs: list = json.load(f)

    mab_state   = load_mab_state(game.mab_state_path)
    mab_updated = False
    updates_made = False

    for run in all_runs:
        if run.get("evaluated", False):
            continue

        target_date = run["target_draw_date"]
        if target_date not in actuals:
            print(f"Run {run['run_id']} → {target_date}: draw not yet available.")
            continue

        actual     = actuals[target_date]
        actual_wbs = actual["wb"]
        actual_mb  = actual["mb"]

        print(f"\n{'='*55}")
        print(f"  Run {run['run_id']}  |  {game.name}  |  Draw {target_date}")
        sb_label = f"  SB:{actual_mb}" if actual_mb is not None else ""
        print(f"  Actual: {sorted(actual_wbs)}{sb_label}")
        print(f"{'='*55}")

        total_spent = 0
        total_won   = 0

        for i, p in enumerate(run["predictions"]):
            mw, mm, won = evaluate_ticket(
                p["wb"], p.get("mb"), actual_wbs, actual_mb, game
            )
            p["match_wbs"] = mw
            p["match_mb"]  = mm
            p["won"]       = won
            total_spent   += TICKET_COST
            total_won     += won

            flag = f"  ⭐ WON ${won:,}" if won > 0 else ""
            sb_str = f" MB:{p.get('mb', '—')}" if game.sb_col else ""
            print(
                f"  T{i+1:02d} [{p.get('strategy','?'):<10}] "
                f"{p['wb']}{sb_str}  →  {mw}+{int(mm)}{flag}"
            )

        roi           = total_won - total_spent
        strat_summary = strategy_summary(run["predictions"])

        print(f"\n  Spent: ${total_spent}  |  Won: ${total_won}  |  ROI: ${roi}")
        print("  Strategy breakdown:")
        for s, d in sorted(strat_summary.items()):
            print(f"    {s}: won ${d['won']} / spent ${d['spent']} = ROI ${d['roi']}")

        run["evaluated"] = True
        run["evaluation_summary"] = {
            "total_spent":  total_spent,
            "total_won":    total_won,
            "net_roi":      roi,
            "by_strategy":  strat_summary,
        }
        updates_made  = True

        # ── v6.0: Update MAB state ────────────────────────────────────────────
        mab_state = update_after_draw(
            mab_state, run["predictions"], total_spent, total_won
        )
        mab_updated = True

    if updates_made:
        with open(game.predictions_path, "w") as f:
            json.dump(all_runs, f, indent=4)
        print(f"\n✅ Results saved to {game.predictions_path}.")

    if mab_updated:
        save_mab_state(mab_state, game.mab_state_path)
        print(f"✅ MAB state updated → {game.mab_state_path}")

    # Always print cumulative + MAB report
    cumulative_report(all_runs, game)
    print_mab_report(mab_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate draw results for any lottery game.")
    parser.add_argument(
        "--game", "-g",
        default="mega_millions",
        choices=list_games(),
        help=f"Which game to evaluate. Default: mega_millions",
    )
    args = parser.parse_args()
    main(get_game(args.game))
