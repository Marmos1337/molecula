"""Record a short video of a real run of the service.

Nothing here is staged: the page talks to the running service, and every number
on screen comes from the saved model.

    make serve        # in one terminal
    PYTHONPATH=src .venv/bin/python scripts/record_demo.py --out reports/demo.mp4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

CIDOFOVIR = "NC1=NC(=O)N(C[C@@H](CO)OCP(=O)(O)O)C=C1"
BATCH = "\n".join(
    [
        CIDOFOVIR,
        "Nc1ncn(C[C@@H](CO)OCP(=O)(O)O)c(=O)n1",
        "CC(=O)Oc1ccccc1C(=O)O",
        "это не молекула",
        "O=c1[nH]c(=O)n([C@H]2C[C@H](O)[C@@H](CO)O2)cc1/C=C\\Br",
    ]
)


def beat(page, seconds: float = 1.2) -> None:
    page.wait_for_timeout(int(seconds * 1000))


def click_if_there(page, selector: str, seconds: float = 1.5) -> bool:
    locator = page.locator(selector).first
    if locator.count() and locator.is_visible():
        locator.click()
        beat(page, seconds)
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", default="reports/demo.mp4")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scratch = out.parent / "_recording"
    shutil.rmtree(scratch, ignore_errors=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(
            viewport={"width": args.width, "height": args.height},
            record_video_dir=str(scratch),
            record_video_size={"width": args.width, "height": args.height},
            locale="ru-RU",
        )
        page = context.new_page()
        page.goto(args.url, wait_until="networkidle")
        page.evaluate("() => localStorage.clear()")
        page.reload(wait_until="networkidle")
        beat(page, 2.0)

        # --- one molecule, start to finish
        page.fill("#smilesInput", CIDOFOVIR)
        beat(page, 1.0)
        page.click("#runSingle")
        page.wait_for_selector("#screen-analysis svg:visible", timeout=60000)
        beat(page, 3.5)
        for _ in range(3):
            page.mouse.wheel(0, 450)
            beat(page, 2.2)
        page.mouse.wheel(0, -1400)
        beat(page, 1.0)

        # --- the collapse control lives in the logo area
        page.hover(".brand")
        beat(page, 1.2)
        page.click("#sidebarToggle")
        beat(page, 1.8)
        page.click("#sidebarToggle")
        beat(page, 1.5)

        # --- the light theme is a full theme, not a tint
        page.click('[data-theme-value="light"]')
        beat(page, 2.5)

        # --- a batch, with one deliberately broken line
        page.click('[data-screen="batch"]')
        beat(page, 1.2)
        page.fill("#batchText", BATCH)
        beat(page, 1.2)
        page.click("#runBatch")
        page.wait_for_selector("#screen-batch tbody tr:visible", timeout=120000)
        beat(page, 4.0)

        # sort by the prediction, then compare two candidates side by side
        header = page.locator("#screen-batch th", has_text="Прогноз").first
        if header.count():
            header.click()
            beat(page, 2.5)

        boxes = page.locator('#screen-batch tbody input[type="checkbox"]')
        if boxes.count() >= 2:
            boxes.nth(0).check()
            boxes.nth(1).check()
            beat(page, 1.2)
            click_if_there(page, 'button:has-text("Сравнить")', 4.0)
            page.keyboard.press("Escape")
            beat(page, 1.0)

        # --- where the numbers come from
        page.click('[data-screen="data"]')
        beat(page, 2.5)
        for _ in range(3):
            page.mouse.wheel(0, 650)
            beat(page, 2.3)

        # --- and the dark theme is complete too
        page.click('[data-theme-value="dark"]')
        beat(page, 3.0)

        video = page.video
        context.close()
        browser.close()
        recorded = Path(video.path()) if video else None

    if not recorded or not recorded.exists():
        print("no video was produced")
        return 1

    if out.suffix == ".mp4" and shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(recorded),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28", "-movflags", "+faststart",
             str(out)],
            check=True,
        )
    else:
        shutil.copy(recorded, out)
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
