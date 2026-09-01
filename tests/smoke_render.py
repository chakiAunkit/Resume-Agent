from pathlib import Path
from playwright.sync_api import sync_playwright
from pypdf import PdfReader

HTML = "<h1>Aunkit Chaki</h1><p>Render smoke test.</p>"
out = Path("output/smoke.pdf")
out.parent.mkdir(exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.set_content(HTML)
    page.pdf(path=str(out), format="A4",
             margin={"top": "0.4in", "bottom": "0.4in",
                     "left": "0.4in", "right": "0.4in"},
             print_background=True)
    browser.close()

print(f"pages: {len(PdfReader(str(out)).pages)}")