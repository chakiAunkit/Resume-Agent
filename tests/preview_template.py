"""Render templates/resume.html.j2 with assets/resume.json -> output/resume.html.

Open the result in Chrome, then Ctrl+P to check layout and page breaks:
  destination "Save as PDF", paper A4, margins "Default",
  "Background graphics" ticked.
Chrome's print engine is the same one Playwright drives headlessly, so
what you see in print preview is what render.py will produce.
"""
from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parents[1]

env = Environment(
    loader=FileSystemLoader(ROOT / "templates"),
    # Autoescape covers .j2 so body text may contain & < > " freely.
    # print.css is included raw (not in the enabled list) on purpose:
    # escaping would mangle CSS child selectors like `.rsection > h2`.
    autoescape=select_autoescape(enabled_extensions=("html", "htm", "xml", "j2")),
    trim_blocks=True,
    lstrip_blocks=True,
)

data = json.loads((ROOT / "assets" / "resume.json").read_text(encoding="utf-8"))
html = env.get_template("resume.html.j2").render(**data, scale=1.0)

out = ROOT / "output" / "resume.html"
out.parent.mkdir(exist_ok=True)
out.write_text(html, encoding="utf-8")
print(f"wrote {out}")
