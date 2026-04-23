#!/usr/bin/env python3
"""
auto_run.py — Waits for the next Mega Millions draw, then runs:
  1. scraper.py    (fetch latest results)
  2. evaluator.py  (score last prediction)
  3. predictor.py  (generate new prediction)

Mega Millions draws: Tuesday & Friday at 11:00 PM ET (10:00 PM CT)
Script waits until 15 min after the draw to ensure results are posted.

Usage:
  python3.13 auto_run.py            # wait for next draw then run
  python3.13 auto_run.py --now      # run immediately (no wait)
  python3.13 auto_run.py --dry-run  # show next draw time, then exit
"""

import subprocess
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ── Config ────────────────────────────────────────────────────────────────────
DRAW_DAYS = {1, 4}          # Monday=0 … Tuesday=1, Friday=4
DRAW_HOUR_ET = 23           # 11:00 PM Eastern Time
DRAW_MINUTE_ET = 0
RESULTS_DELAY_MINUTES = 15  # wait this long after draw before scraping

EASTERN = ZoneInfo("America/New_York")
PROJECT_DIR = "/Users/jiawenzhu/megamillions-engine"
PYTHON = "python3.13"

STEPS = [
    ("scraper.py",   "Fetching latest Mega Millions results"),
    ("evaluator.py", "Evaluating last prediction"),
    ("predictor.py", "Generating new prediction"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def now_et() -> datetime:
    return datetime.now(tz=EASTERN)

def next_draw_after(dt: datetime) -> datetime:
    """Return the next draw datetime (ET) strictly after `dt`."""
    candidate = dt.replace(
        hour=DRAW_HOUR_ET,
        minute=DRAW_MINUTE_ET,
        second=0,
        microsecond=0,
    )
    # Advance day by day until we land on a draw day
    for _ in range(7):
        if candidate > dt and candidate.weekday() in DRAW_DAYS:
            return candidate
        candidate += timedelta(days=1)
    raise RuntimeError("Could not find next draw day within 7 days")

def fmt_duration(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)

def section(title: str):
    bar = "─" * 50
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")

def run_step(script: str, label: str) -> bool:
    """Run one Python script, streaming output. Returns True on success."""
    section(label)
    print(f"  Running: {PYTHON} {script}\n")
    result = subprocess.run(
        [PYTHON, script],
        cwd=PROJECT_DIR,
    )
    if result.returncode != 0:
        print(f"\n  ERROR: {script} exited with code {result.returncode}")
        return False
    print(f"\n  Done: {script}")
    return True

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    run_now = "--now" in sys.argv
    dry_run = "--dry-run" in sys.argv

    print("╔══════════════════════════════════════════════════════╗")
    print("║          Mega Millions Auto-Run Engine               ║")
    print("╚══════════════════════════════════════════════════════╝")

    if run_now:
        print("\n  Mode: IMMEDIATE (--now flag detected)\n")
    else:
        current = now_et()
        draw_time = next_draw_after(current)
        run_time = draw_time + timedelta(minutes=RESULTS_DELAY_MINUTES)
        wait_secs = (run_time - current).total_seconds()

        day_name = draw_time.strftime("%A")
        draw_str = draw_time.strftime("%Y-%m-%d %I:%M %p ET")
        run_str = run_time.strftime("%Y-%m-%d %I:%M %p ET")

        print(f"\n  Next draw  :  {day_name}, {draw_str}")
        print(f"  Scripts run:  {run_str}  (+{RESULTS_DELAY_MINUTES} min for results to post)")
        print(f"  Wait time  :  {fmt_duration(wait_secs)}")

        if dry_run:
            print("\n  Dry-run mode — exiting without waiting.\n")
            return

        print("\n  Waiting... (Ctrl+C to cancel)\n")
        try:
            # Sleep in chunks so we can print a countdown every 30 min
            INTERVAL = 30 * 60
            while wait_secs > 0:
                chunk = min(INTERVAL, wait_secs)
                time.sleep(chunk)
                wait_secs -= chunk
                if wait_secs > 0:
                    print(f"  Still waiting: {fmt_duration(wait_secs)} remaining...")
        except KeyboardInterrupt:
            print("\n\n  Cancelled by user.\n")
            sys.exit(0)

        print(f"\n  Draw time reached — starting pipeline...\n")

    # ── Run the pipeline ──────────────────────────────────────────────────────
    success = True
    for script, label in STEPS:
        ok = run_step(script, label)
        if not ok:
            print(f"\n  Pipeline aborted at {script}. Fix the error above and re-run.\n")
            success = False
            break

    print("\n" + "═" * 54)
    if success:
        print("  All steps completed successfully!")
    else:
        print("  Pipeline finished with errors.")
    print("═" * 54 + "\n")

if __name__ == "__main__":
    main()
