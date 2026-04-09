"""
predictor.py – Multi-Game Quantitative Predictor v6.0
======================================================
Upgrades from v4.0:
  - Supports any game via --game argument (mega_millions / powerball / lotto)
  - Uses data-driven RejectionFilters loaded from calibration JSON (v6.0 filters.py)
  - Falls back to mathematical defaults if calibration hasn't been run yet.
  - Tags predictions with "source": "ensemble_v6"

Each ticket is drawn from the ENSEMBLE distribution (all strategies vote),
so there are no longer purely "hot" or "cold" tickets — every ticket
reflects the combined wisdom of all strategies, weighted by their track record.
"""

import argparse
import json
import os
import random
import uuid
from datetime import timedelta

import pandas as pd

from games import get_game, list_games
from games.base_game import BaseGame
from filters import RejectionFilters
from utils import load_history, calculate_frequency, calculate_overdue
from mab_engine import (
    load_mab_state,
    save_mab_state,
    kelly_budget,
    allocate_tickets,
    ensemble_wb_probs,
    ensemble_mb_probs,
    print_mab_report,
    STRATEGIES,
    TICKET_COST,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_next_draw_date(latest_draw_date: pd.Timestamp, game: BaseGame) -> str:
    """Return the next draw date for the given game."""
    current = latest_draw_date + timedelta(days=1)
    while current.weekday() not in game.draw_days:
        current += timedelta(days=1)
    return current.strftime("%Y-%m-%d")


def weighted_sample_no_replace(population: list, weights: list, k: int) -> list:
    """Weighted sampling WITHOUT replacement."""
    result = []
    pop    = list(population)
    wts    = list(weights)
    for _ in range(k):
        total  = sum(wts)
        r      = random.uniform(0, total)
        upto   = 0.0
        chosen = len(pop) - 1
        for i, w in enumerate(wts):
            upto += w
            if upto >= r:
                chosen = i
                break
        result.append(pop.pop(chosen))
        wts.pop(chosen)
    return sorted(result)


# ── Strategy Weight Builders ──────────────────────────────────────────────────

def build_hot_weights(wb_counts, sb_counts, game: BaseGame):
    wb_w = {i: max(1, wb_counts.get(i, 0) * 3) for i in game.all_wb_numbers()}
    sb_w = {i: max(1, sb_counts.get(i, 0) * 3) for i in game.all_sb_numbers()} if game.sb_col else {}
    return wb_w, sb_w


def build_cold_weights(wb_last_seen, sb_last_seen, game: BaseGame):
    wb_w = {i: max(1, wb_last_seen.get(i, 999) / 5.0) for i in game.all_wb_numbers()}
    sb_w = {i: max(1, sb_last_seen.get(i, 999) / 5.0) for i in game.all_sb_numbers()} if game.sb_col else {}
    return wb_w, sb_w


def build_hybrid_weights(wb_counts, sb_counts, wb_last_seen, sb_last_seen, game: BaseGame):
    wb_w = {
        i: max(1, wb_counts.get(i, 0)) * max(1, wb_last_seen.get(i, 999) / 5.0)
        for i in game.all_wb_numbers()
    }
    sb_w = {
        i: max(1, sb_counts.get(i, 0)) * max(1, sb_last_seen.get(i, 999) / 5.0)
        for i in game.all_sb_numbers()
    } if game.sb_col else {}
    return wb_w, sb_w


def build_uniform_weights(game: BaseGame):
    return {i: 1 for i in game.all_wb_numbers()}, {i: 1 for i in game.all_sb_numbers()}


# ── Ensemble Ticket Drawing with v6 Rejection Sampling ───────────────────────

def draw_ensemble_ticket(
    wb_prob_map: dict[int, float],
    sb_prob_map: dict[int, float],
    game: BaseGame,
    filters: RejectionFilters,
) -> dict:
    """
    Draw one ticket using rejection sampling.
    Tries up to 10,000 times to find a combination that passes all filters.
    Falls back to unfiltered draw if needed.
    """
    nums    = list(wb_prob_map.keys())
    wts     = list(wb_prob_map.values())
    mid     = (game.wb_range[0] + game.wb_range[1]) / 2.0

    # Special ball
    has_sb = bool(game.sb_col) and sb_prob_map
    sb_nums = list(sb_prob_map.keys()) if has_sb else []
    sb_wts  = list(sb_prob_map.values()) if has_sb else []

    for _ in range(10_000):
        wb_draw = weighted_sample_no_replace(nums, wts, game.wb_count)
        if filters.evaluate_all(wb_draw, mid):
            sb_draw = random.choices(sb_nums, weights=sb_wts, k=1)[0] if has_sb else None
            return {"wb": wb_draw, "mb": sb_draw}

    # Fallback (no filter)
    wb_draw = weighted_sample_no_replace(nums, wts, game.wb_count)
    sb_draw = random.choices(sb_nums, weights=sb_wts, k=1)[0] if has_sb else None
    return {"wb": wb_draw, "mb": sb_draw}


# ── Main ──────────────────────────────────────────────────────────────────────

def main(game: BaseGame) -> None:
    print(f"\n🎰 Mega Millions Engine v6.0 — {game.name}")
    print(f"   CSV          : {game.csv_path}")
    print(f"   Predictions  : {game.predictions_path}")
    print(f"   MAB state    : {game.mab_state_path}")

    # ── Load filters ─────────────────────────────────────────────────────────
    try:
        filters = RejectionFilters.from_game(game)
        print(f"   Filters (cal): {filters}")
    except FileNotFoundError:
        filters = RejectionFilters.default_fallback(game)
        print(f"   Filters (est): {filters}")
        print("   ⚠️  No calibration file. Run: python3.13 calibrate.py --game " + game.slug)

    # ── Load historical draw data ─────────────────────────────────────────────
    try:
        df = load_history(game)
    except FileNotFoundError:
        print(f"Error: {game.csv_path} not found. Run scraper.py --game {game.slug} first.")
        return

    if df.empty:
        print("No data found.")
        return

    latest_draw_date = df["DrawDate"].iloc[0]
    next_draw_date   = get_next_draw_date(latest_draw_date, game)
    print(f"\n   Latest real draw   : {latest_draw_date.strftime('%Y-%m-%d')}")
    print(f"   Predicting for     : {next_draw_date}")
    print(f"   Historical records : {len(df)} draws")

    # ── Load existing prediction runs ─────────────────────────────────────────
    os.makedirs(os.path.dirname(game.predictions_path) if os.path.dirname(game.predictions_path) else ".", exist_ok=True)
    all_runs: list = []
    if os.path.exists(game.predictions_path):
        with open(game.predictions_path, "r") as f:
            try:
                all_runs = json.load(f)
            except json.JSONDecodeError:
                pass

    # ── Load MAB state ────────────────────────────────────────────────────────
    mab_state = load_mab_state(game.mab_state_path)
    is_cold_start = mab_state["draw_count"] == 0

    # ── Kelly Criterion + Thompson Sampling ───────────────────────────────────
    budget, thetas, kelly_fracs = kelly_budget(mab_state)
    allocation = allocate_tickets(budget, thetas)
    print_mab_report(mab_state, thetas, kelly_fracs)

    if is_cold_start:
        print("\n  ⚠️  Cold start: MAB state initialised with uniform priors.")

    print(f"\n  💰 Kelly Budget : {budget} tickets  (virtual spend: ${budget * TICKET_COST})")
    print(f"  🏦 Bankroll     : ${mab_state['virtual_bankroll']:,.2f}\n")

    # ── Build per-strategy weight dicts ───────────────────────────────────────
    wb_counts, sb_counts       = calculate_frequency(df, game)
    wb_last_seen, sb_last_seen = calculate_overdue(df, game)

    all_wb_weights = {
        "A-hot":    build_hot_weights(wb_counts, sb_counts, game)[0],
        "B-cold":   build_cold_weights(wb_last_seen, sb_last_seen, game)[0],
        "D-hybrid": build_hybrid_weights(wb_counts, sb_counts, wb_last_seen, sb_last_seen, game)[0],
        "C-random": build_uniform_weights(game)[0],
    }
    all_sb_weights = {
        "A-hot":    build_hot_weights(wb_counts, sb_counts, game)[1],
        "B-cold":   build_cold_weights(wb_last_seen, sb_last_seen, game)[1],
        "D-hybrid": build_hybrid_weights(wb_counts, sb_counts, wb_last_seen, sb_last_seen, game)[1],
        "C-random": build_uniform_weights(game)[1],
    }

    # ── Build ENSEMBLE probability maps ───────────────────────────────────────
    wb_ensemble = ensemble_wb_probs(mab_state, all_wb_weights)
    sb_ensemble = ensemble_mb_probs(mab_state, all_sb_weights) if game.sb_col else {}

    # ── Print allocation ──────────────────────────────────────────────────────
    print(f"  {'Strategy':<12} {'θ (TS)':>8} {'Kelly f*':>9} {'Tickets':>8}  EMA ROI")
    print("  " + "─" * 50)
    for s in STRATEGIES:
        ema = mab_state["strategies"][s]["ema_roi"]
        print(
            f"  {s:<12} {thetas[s]:>8.4f} {kelly_fracs[s]:>9.4f} "
            f"{allocation[s]:>8}  {ema:+.4f}"
        )
    print()

    # ── Draw tickets from ensemble ────────────────────────────────────────────
    predictions = []
    print(f"  {'Ticket':<6} {'Label':<12} {'Numbers':<36} SB")
    print("  " + "─" * 60)

    ticket_num = 1
    for strategy, n_tickets in allocation.items():
        for _ in range(n_tickets):
            ticket = draw_ensemble_ticket(wb_ensemble, sb_ensemble, game, filters)
            ticket["strategy"] = strategy
            ticket["source"]   = "ensemble_v6"
            predictions.append(ticket)

            sb_str = str(ticket["mb"]) if ticket["mb"] is not None else "—"
            wb_str = str(ticket["wb"]).ljust(35)
            print(f"  {ticket_num:02d}     [{strategy:<10}] {wb_str} {sb_str}")
            ticket_num += 1

    # ── Capture ensemble weights snapshot ─────────────────────────────────────
    ensemble_snapshot = {
        s: {
            "theta":      round(thetas[s], 6),
            "kelly_f":    round(kelly_fracs[s], 6),
            "ema_roi":    round(mab_state["strategies"][s]["ema_roi"], 6),
            "allocation": allocation[s],
        }
        for s in STRATEGIES
    }

    # ── Save predictions ──────────────────────────────────────────────────────
    run_record = {
        "run_id":           str(uuid.uuid4())[:8],
        "version":          "6.0",
        "game":             game.slug,
        "created_at":       pd.Timestamp.now().isoformat(),
        "target_draw_date": next_draw_date,
        "history_size":     len(df),
        "budget":           budget,
        "bankroll_at_pred": mab_state["virtual_bankroll"],
        "allocation":       allocation,
        "ensemble_weights": ensemble_snapshot,
        "mab_draw_count":   mab_state["draw_count"],
        "predictions":      predictions,
        "evaluated":        False,
        "evaluation_summary": None,
    }
    all_runs.append(run_record)

    with open(game.predictions_path, "w") as f:
        json.dump(all_runs, f, indent=4)

    save_mab_state(mab_state, game.mab_state_path)

    print(
        f"\n✅ {len(predictions)} ensemble tickets saved to {game.predictions_path}."
        f"\n   Run evaluator.py --game {game.slug} after the {next_draw_date} draw!"
    )

    # ── Auto-sync to Google Sheets (Mega Millions only for now) ───────────────
    if game.slug == "mega_millions":
        try:
            from sheets_sync import sync_predictions
            sync_predictions(run_id=run_record["run_id"])
        except Exception as e:
            print(f"\n   [sheets_sync] skipped: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate lottery ticket predictions.")
    parser.add_argument(
        "--game", "-g",
        default="mega_millions",
        choices=list_games(),
        help=f"Which game to predict. Default: mega_millions",
    )
    args = parser.parse_args()
    main(get_game(args.game))
