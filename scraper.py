"""
scraper.py — Multi-Game History Scraper v6.0
=============================================
Supports any game registered in games/__init__.py.

Usage:
    python3.13 scraper.py                       # defaults to mega_millions
    python3.13 scraper.py --game mega_millions
    python3.13 scraper.py --game powerball
    python3.13 scraper.py --game lotto
"""

import argparse
import asyncio
import csv
import os

from playwright.async_api import async_playwright

from games import get_game, list_games
from games.base_game import BaseGame


def load_existing_dates(csv_path: str) -> set:
    """Return a set of date strings already in the CSV."""
    if not os.path.exists(csv_path):
        return set()
    existing = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("DrawDate"):
                existing.add(row["DrawDate"].strip())
    return existing


def parse_row(columns_text: list[str], game: BaseGame) -> list[str] | None:
    """
    Parse a list of TD text values into a flat row.
    Returns [date, WB1..WBn, SB (if applicable), Extra] or None.
    """
    if len(columns_text) < 3:
        return None

    date_text = columns_text[0].strip()
    winning_text = columns_text[1].strip()
    sb_text = columns_text[2].strip() if game.sb_col else ""
    extra = columns_text[3].strip() if len(columns_text) >= 4 else ""

    white_balls = [b.strip() for b in winning_text.split("-")]
    if len(white_balls) != game.wb_count:
        return None

    if game.sb_col:
        return [date_text] + white_balls + [sb_text, extra]
    else:
        return [date_text] + white_balls + [extra]


async def scrape_page(page, url: str, game: BaseGame) -> list[list[str]]:
    """Scrape a single results page and return raw row data."""
    await page.goto(url, wait_until="domcontentloaded")
    rows = page.locator(game.scrape_table_selector)
    count = await rows.count()

    data = []
    for i in range(count):
        row = rows.nth(i)
        columns = row.locator("td")
        num_cols = await columns.count()
        if num_cols < 3:
            continue
        cols_text = [await columns.nth(j).inner_text() for j in range(num_cols)]
        parsed = parse_row(cols_text, game)
        if parsed:
            data.append(parsed)
    return data


async def scrape_game(game: BaseGame):
    """Scrape historical data for a single game and append to its CSV."""
    csv_path = game.csv_path
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)

    existing_dates = load_existing_dates(csv_path)
    print(f"\n🎰 Scraping {game.name}")
    print(f"   CSV target    : {csv_path}")
    print(f"   Existing rows : {len(existing_dates)}")
    print(f"   Source URL    : {game.scrape_url}")

    all_new_rows = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page_obj = await browser.new_page()

        for page_num in range(1, 30):  # max 30 pages
            url = game.scrape_url if page_num == 1 else f"{game.scrape_url}?pagenum={page_num}"
            print(f"\n   Fetching page {page_num}: {url}")
            rows = await scrape_page(page_obj, url, game)

            if not rows:
                print(f"   No data found on page {page_num}, stopping.")
                break

            new_on_page = 0
            for row in rows:
                date_str = row[0]
                if date_str not in existing_dates:
                    all_new_rows.append(row)
                    existing_dates.add(date_str)
                    new_on_page += 1

            print(f"   Found {len(rows)} rows, {new_on_page} new.")
            if new_on_page == 0:
                print("   All rows on this page already in CSV, stopping.")
                break

        await browser.close()

    if not all_new_rows:
        print(f"\n✅ {game.name}: CSV is up to date. No new draws found.")
        return

    # Build header
    header = ["DrawDate"] + game.wb_cols
    if game.sb_col:
        header.append(game.sb_col)
    header.append("Multiplier")

    file_exists = os.path.exists(csv_path)
    write_mode = "a" if file_exists else "w"

    with open(csv_path, mode=write_mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerows(all_new_rows)

    print(f"\n✅ Added {len(all_new_rows)} new draws to {csv_path}.")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape lottery history for any supported game."
    )
    parser.add_argument(
        "--game", "-g",
        default="mega_millions",
        choices=list_games(),
        help=f"Which game to scrape. Choices: {', '.join(list_games())}",
    )
    args = parser.parse_args()

    game = get_game(args.game)
    asyncio.run(scrape_game(game))


if __name__ == "__main__":
    main()
