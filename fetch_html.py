import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        # Launch Chromium headless
        browser = await p.chromium.launch()
        page = await browser.new_page()
        print("Navigating to page...")
        await page.goto("https://www.illinoislottery.com/dbg/results/megamillions?page=1", wait_until="domcontentloaded")
        await page.wait_for_timeout(5000)
        
        # Save HTML for inspection
        html = await page.content()
        with open("/tmp/megamillions.html", "w") as f:
            f.write(html)
        print("HTML saved to /tmp/megamillions.html")
        await browser.close()

asyncio.run(run())
