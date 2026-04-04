"""
sheets_sync.py — Sync Mega Millions predictions to Google Sheets.

Usage (standalone):
    python3.13 sheets_sync.py                   # sync all runs in predictions.json
    python3.13 sheets_sync.py --run-id <id>     # sync a single run

Called automatically by predictor.py after saving predictions.json.

Authentication:
    First run opens a browser OAuth flow and caches credentials in
    ~/.cache/megamillions_token.json  (only needed once).
    Uses your Google account — no service account / JSON key required.

Sheet layout (one sheet = "Predictions"):
    RunID | CreatedAt | TargetDraw | Strategy | Ticket# | N1 N2 N3 N4 N5 | MB | Spent | Won | ROI
"""

import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

try:
    import gspread
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
except ImportError:
    print("❌  Missing dependencies — run:")
    print("    pip3.13 install gspread google-auth-oauthlib --break-system-packages")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
PREDICTIONS_FILE = Path(__file__).parent / "predictions.json"
TOKEN_FILE       = Path.home() / ".cache" / "megamillions_token.json"
CLIENT_ID_FILE   = Path(__file__).parent / "google_oauth_client.json"

SPREADSHEET_NAME = "Mega Millions Predictions"
SHEET_NAME       = "Predictions"
SCOPES           = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]

HEADER = [
    "RunID", "CreatedAt", "TargetDraw",
    "Strategy", "Ticket#",
    "N1", "N2", "N3", "N4", "N5", "MegaBall",
    "Budget", "Spent($)", "Won($)", "NetROI($)",
    "theta_A", "theta_B", "theta_C", "theta_D",
]

# ── Auth ──────────────────────────────────────────────────────────────────────

def _get_credentials():
    """Return valid Google API credentials, refreshing or re-authenticating as needed."""
    creds = None
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_token(creds)
            return creds
        except Exception:
            pass  # fall through to re-auth

    # Need full OAuth flow
    if not CLIENT_ID_FILE.exists():
        print(f"\n⚠️  No OAuth client file found at: {CLIENT_ID_FILE}")
        print("   To set up Google Sheets sync, follow these steps:\n")
        print("   1. Go to https://console.cloud.google.com/apis/credentials")
        print("   2. Create → OAuth 2.0 Client ID → Desktop App")
        print("   3. Download JSON → save as:  google_oauth_client.json")
        print("      (in the megamillions-engine/ folder)")
        print("   4. Enable Google Sheets API + Google Drive API for your project")
        print("   5. Re-run predictor.py\n")
        return None

    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_ID_FILE), SCOPES)
    creds = flow.run_local_server(port=0, open_browser=True)
    _save_token(creds)
    return creds


def _save_token(creds):
    TOKEN_FILE.write_text(creds.to_json())


# ── Sheet helpers ─────────────────────────────────────────────────────────────

def _open_or_create_sheet(gc):
    """Open the spreadsheet by name, creating it if needed."""
    try:
        sh = gc.open(SPREADSHEET_NAME)
    except gspread.SpreadsheetNotFound:
        sh = gc.create(SPREADSHEET_NAME)
        sh.share(None, perm_type="anyone", role="writer")  # make it accessible
        print(f"   📊 Created new spreadsheet: {SPREADSHEET_NAME}")

    try:
        ws = sh.worksheet(SHEET_NAME)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(SHEET_NAME, rows=5000, cols=len(HEADER))
        ws.append_row(HEADER, value_input_option="USER_ENTERED")
        # Bold the header
        ws.format("A1:S1", {"textFormat": {"bold": True}})
        print(f"   📋 Created sheet '{SHEET_NAME}' with header row")

    return sh, ws


def _existing_run_ids(ws):
    """Return a set of RunIDs already written to the sheet (avoids duplicates)."""
    try:
        col = ws.col_values(1)  # RunID column
        return set(col[1:])     # skip header
    except Exception:
        return set()


# ── Core sync ─────────────────────────────────────────────────────────────────

def _run_to_rows(run: dict) -> list[list]:
    """Convert a single prediction run dict into a list of sheet rows (one per ticket)."""
    run_id     = run.get("run_id", "?")
    created_at = run.get("created_at", "")[:19]   # trim microseconds
    target     = run.get("target_draw_date", "")
    budget     = run.get("budget", 0)

    # evaluation data (filled by evaluator.py later)
    summary    = run.get("evaluation_summary") or {}
    spent      = summary.get("total_spent", budget * 2)
    won        = summary.get("total_won", 0)
    roi        = won - spent

    weights    = run.get("ensemble_weights", {})
    theta      = {k: round(v.get("theta", 0), 4) for k, v in weights.items()}

    rows = []
    for idx, ticket in enumerate(run.get("predictions", []), start=1):
        wb        = ticket.get("wb", [])      # white balls list
        strategy  = ticket.get("strategy", "?")
        mb        = ticket.get("mb", "?")

        n1, n2, n3, n4, n5 = (list(wb) + ["", "", "", "", ""])[:5]

        row = [
            run_id, created_at, target,
            strategy, idx,
            n1, n2, n3, n4, n5, mb,
            budget, spent, won, roi,
            theta.get("A-hot",   ""),
            theta.get("B-cold",  ""),
            theta.get("C-random",""),
            theta.get("D-hybrid",""),
        ]
        rows.append(row)

    return rows


def sync_predictions(run_id: str | None = None, verbose: bool = True) -> bool:
    """
    Sync predictions.json to Google Sheets.

    Args:
        run_id: If given, only sync that specific run. Otherwise sync all new runs.
        verbose: Print progress to stdout.
    Returns:
        True if sync succeeded, False on any error (non-fatal — predictor continues).
    """
    if verbose:
        print("\n📊 Syncing to Google Sheets …")

    # Load predictions
    if not PREDICTIONS_FILE.exists():
        if verbose:
            print("   ⚠️  predictions.json not found — skipping sync")
        return False

    with open(PREDICTIONS_FILE) as f:
        data = json.load(f)

    # Normalise to list
    runs = data if isinstance(data, list) else list(data.values())

    if run_id:
        runs = [r for r in runs if r.get("run_id") == run_id]
        if not runs:
            if verbose:
                print(f"   ⚠️  Run {run_id} not found in predictions.json")
            return False

    # Authenticate
    creds = _get_credentials()
    if creds is None:
        return False  # setup instructions already printed

    gc = gspread.authorize(creds)
    sh, ws = _open_or_create_sheet(gc)

    existing = _existing_run_ids(ws)
    new_runs = [r for r in runs if r.get("run_id") not in existing]

    if not new_runs:
        if verbose:
            print("   ✅  Sheet already up to date — nothing new to add")
        return True

    # Batch-append all rows at once (much faster than one-by-one)
    all_rows = []
    for run in new_runs:
        all_rows.extend(_run_to_rows(run))

    ws.append_rows(all_rows, value_input_option="USER_ENTERED")

    url = f"https://docs.google.com/spreadsheets/d/{sh.id}"
    if verbose:
        print(f"   ✅  Added {len(all_rows)} ticket rows ({len(new_runs)} run(s))")
        print(f"   🔗  {url}")

    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Mega Millions predictions → Google Sheets")
    parser.add_argument("--run-id", help="Sync only this specific run ID")
    args = parser.parse_args()

    ok = sync_predictions(run_id=args.run_id)
    sys.exit(0 if ok else 1)
