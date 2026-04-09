"""
calibrate.py — Data-Driven Statistical Calibration for Lottery Engine v6.0
===========================================================================
Reads a game's historical CSV and computes precise statistical boundaries
that filters.py will use instead of hardcoded guesses.

Usage:
    python3.13 calibrate.py --game mega_millions
    python3.13 calibrate.py --game powerball
    python3.13 calibrate.py --game lotto

Output:
    data/{slug}_calibration.json (auto-created, overwritten each time)
"""

import argparse
import json
import math
import os

import pandas as pd

from games import get_game


def compute_percentile(values: list[float], p: float) -> float:
    """Simple percentile computation (no numpy required)."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (p / 100.0) * (len(sorted_v) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_v) - 1)
    frac = idx - lo
    return sorted_v[lo] + frac * (sorted_v[hi] - sorted_v[lo])


def calibrate(game) -> dict:
    """
    Read the game's CSV and compute all statistical boundaries needed by filters.py.
    Returns a calibration dict that is also written to disk.
    """
    csv_path = game.csv_path

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No history found at '{csv_path}'. "
            f"Run: python3.13 scraper.py --game {game.slug}"
        )

    df = pd.read_csv(csv_path)
    df["DrawDate"] = pd.to_datetime(df["DrawDate"], format="mixed", dayfirst=False)
    n = len(df)
    print(f"\n📊 Calibrating {game.name} from {n} draws ({csv_path})")

    wb_cols = game.wb_cols

    # ── 1. Sum statistics ────────────────────────────────────────────────────
    sums = df[wb_cols].sum(axis=1).dropna().tolist()
    sum_mean = sum(sums) / len(sums)
    sum_var  = sum((x - sum_mean) ** 2 for x in sums) / len(sums)
    sum_std  = math.sqrt(sum_var)

    sum_low_1s  = sum_mean - 1.0 * sum_std
    sum_high_1s = sum_mean + 1.0 * sum_std
    sum_low_15s = sum_mean - 1.5 * sum_std
    sum_high_15s = sum_mean + 1.5 * sum_std
    sum_low_2s  = sum_mean - 2.0 * sum_std
    sum_high_2s = sum_mean + 2.0 * sum_std

    # ── 2. Gap statistics (adjacent sorted ball differences) ─────────────────
    all_max_gaps = []
    all_min_gaps = []
    for _, row in df.iterrows():
        vals = sorted(int(row[c]) for c in wb_cols if pd.notna(row[c]))
        if len(vals) == game.wb_count:
            gaps = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
            all_max_gaps.append(max(gaps))
            all_min_gaps.append(min(gaps))

    gap_p90 = compute_percentile(all_max_gaps, 90)
    gap_p95 = compute_percentile(all_max_gaps, 95)
    gap_p99 = compute_percentile(all_max_gaps, 99)

    # ── 3. Parity distribution (even count per draw) ────────────────────────
    even_counts = []
    for _, row in df.iterrows():
        vals = [int(row[c]) for c in wb_cols if pd.notna(row[c])]
        even_counts.append(sum(1 for v in vals if v % 2 == 0))

    parity_dist = {}
    for k in range(game.wb_count + 1):
        parity_dist[str(k)] = round(even_counts.count(k) / len(even_counts), 4)

    # Safe even range: include any parity count that appears in >2% of draws
    safe_even_min = min(int(k) for k, v in parity_dist.items() if float(v) > 0.02)
    safe_even_max = max(int(k) for k, v in parity_dist.items() if float(v) > 0.02)

    # ── 4. High/Low distribution ─────────────────────────────────────────────
    mid = (game.wb_range[0] + game.wb_range[1]) / 2.0
    high_counts = []
    for _, row in df.iterrows():
        vals = [int(row[c]) for c in wb_cols if pd.notna(row[c])]
        high_counts.append(sum(1 for v in vals if v > mid))

    high_dist = {}
    for k in range(game.wb_count + 1):
        high_dist[str(k)] = round(high_counts.count(k) / len(high_counts), 4)

    safe_high_min = min(int(k) for k, v in high_dist.items() if float(v) > 0.02)
    safe_high_max = max(int(k) for k, v in high_dist.items() if float(v) > 0.02)

    # ── Build calibration dict ────────────────────────────────────────────────
    calibration = {
        "game":  game.slug,
        "draws": n,
        "sum": {
            "mean": round(sum_mean, 2),
            "std":  round(sum_std, 2),
            "low_1sigma":  round(sum_low_1s, 1),
            "high_1sigma": round(sum_high_1s, 1),
            "low_1_5sigma":  round(sum_low_15s, 1),
            "high_1_5sigma": round(sum_high_15s, 1),
            "low_2sigma":  round(sum_low_2s, 1),
            "high_2sigma": round(sum_high_2s, 1),
        },
        "gap": {
            "max_p90": round(gap_p90, 1),
            "max_p95": round(gap_p95, 1),
            "max_p99": round(gap_p99, 1),
        },
        "parity": {
            "distribution": parity_dist,
            "safe_even_min": safe_even_min,
            "safe_even_max": safe_even_max,
        },
        "high_low": {
            "midpoint":       mid,
            "distribution":   high_dist,
            "safe_high_min":  safe_high_min,
            "safe_high_max":  safe_high_max,
        },
    }

    # ── Print readable report ────────────────────────────────────────────────
    print(f"\n  {'─'*55}")
    print(f"  Sum  :  mean={sum_mean:.1f}  σ={sum_std:.1f}")
    print(f"         ±1σ  [{sum_low_1s:.0f}, {sum_high_1s:.0f}]")
    print(f"         ±1.5σ [{sum_low_15s:.0f}, {sum_high_15s:.0f}]  ← default filter")
    print(f"         ±2σ  [{sum_low_2s:.0f}, {sum_high_2s:.0f}]")
    print(f"\n  Max Gap:  p90={gap_p90:.0f}, p95={gap_p95:.0f}, p99={gap_p99:.0f}")
    print(f"\n  Parity (even count distribution):")
    for k, v in parity_dist.items():
        bar = "█" * int(v * 40)
        print(f"    {k} even: {v:.1%}  {bar}")
    print(f"  → Safe even range: [{safe_even_min}, {safe_even_max}]")
    print(f"\n  High/Low (balls > {mid}):")
    for k, v in high_dist.items():
        bar = "█" * int(v * 40)
        print(f"    {k} high: {v:.1%}  {bar}")
    print(f"  → Safe high range: [{safe_high_min}, {safe_high_max}]")

    # ── Save to disk ─────────────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    out_path = game.calibration_path
    with open(out_path, "w") as f:
        json.dump(calibration, f, indent=4)
    print(f"\n✅ Calibration saved to {out_path}")

    return calibration


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate statistical filters from actual historical data."
    )
    parser.add_argument(
        "--game", "-g",
        default="mega_millions",
        help="Game to calibrate (mega_millions, powerball, lotto). Default: mega_millions",
    )
    args = parser.parse_args()

    game = get_game(args.game)
    calibrate(game)


if __name__ == "__main__":
    main()
