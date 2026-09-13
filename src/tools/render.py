"""Deterministic render tool: resume data -> validated HTML -> PDF -> page count.

Roadmap invariant #2 lives here: the page count is *measured* by pypdf
from the actual artifact, never estimated by a model. Downstream, the
fit loop and the Judge both read this number instead of asking an LLM
"does this look like one page?".

Phase 3 addition: `RenderResult.fill` — how many pages' worth of content
the document holds, as a float (0.92 = fits one page with 8% spare,
1.04 = one page plus a few lines). Measured from the laid-out HTML in
print media at the paper's content width, so it's an estimate of the
same layout Chromium prints; the page count from pypdf stays the hard
truth. The fit loop uses `fill` to choose a step size and `page_count`
to decide whether it's done.

Layering:
    build_html()    pure function, no I/O   — schema validation + Jinja
    Renderer        reusable Chromium session — launch once, render many
    render_resume() one-shot convenience     — the roadmap signature
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from playwright.sync_api import sync_playwright
from pypdf import PdfReader

from src.models import Resume

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = ROOT / "templates"
DEFAULT_OUT = ROOT / "output" / "resume.pdf"

# Canonical bounds for the fit loop's knob. 0.80 puts the base font at
# ~8.4pt — the readability floor. fit.py imports these instead of
# inventing its own.
SCALE_MIN = 0.80
SCALE_MAX = 1.20

# Paper geometry in CSS pixels (96 dpi), mirroring @page in print.css:
# A4 = 210 x 297 mm = 794 x 1123 px; margin 0.35in = 33.6px each side.
# print.css stays the source of truth for *printing*; these are only
# used to measure fill, and must be changed together with it.
_PAGE_W, _PAGE_H, _MARGIN = 794, 1123, 33.6
CONTENT_WIDTH_PX = int(_PAGE_W - 2 * _MARGIN)   # 726
CONTENT_HEIGHT_PX = _PAGE_H - 2 * _MARGIN       # 1055.8

_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    # .j2 is autoescaped (content may contain & < > "); print.css is
    # deliberately excluded so CSS selectors pass through raw.
    autoescape=select_autoescape(enabled_extensions=("html", "htm", "xml", "j2")),
    trim_blocks=True,
    lstrip_blocks=True,
)


def build_html(data: dict | Resume, scale: float = 1.0) -> str:
    """Validate data against the schema and render the HTML string.

    Pure function: raises fast on bad input (pydantic ValidationError,
    ValueError on scale) *before* any browser work happens.
    """
    if not SCALE_MIN <= scale <= SCALE_MAX:
        raise ValueError(
            f"scale={scale} outside [{SCALE_MIN}, {SCALE_MAX}] — "
            "below the floor the resume becomes unreadable; fix content instead."
        )
    resume = data if isinstance(data, Resume) else Resume.model_validate(data)
    return _env.get_template("resume.html.j2").render(
        **resume.render_context(), scale=scale
    )


@dataclass(frozen=True)
class RenderResult:
    pdf_path: Path
    html_path: Path  # debug artifact: open in a browser when layout looks wrong
    page_count: int  # measured by pypdf — the hard constraint
    fill: float      # pages' worth of content, measured from layout — the soft signal
    scale: float


class Renderer:
    """Reusable Chromium session.

    Browser startup (~1s) dominates render time, and the fit loop renders
    up to 4 times per job — so hold one browser open and reuse it:

        with Renderer() as r:
            result = r.render(data, scale=0.9)
    """

    def __enter__(self) -> "Renderer":
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch()
        return self

    def __exit__(self, *exc) -> None:
        self._browser.close()
        self._pw.stop()

    def render(
        self,
        data: dict | Resume,
        scale: float = 1.0,
        out_path: Path | str | None = None,
    ) -> RenderResult:
        out = Path(out_path) if out_path else DEFAULT_OUT
        out.parent.mkdir(parents=True, exist_ok=True)

        html = build_html(data, scale)
        html_path = out.with_suffix(".html")
        html_path.write_text(html, encoding="utf-8")

        page = self._browser.new_page()
        try:
            # Lay the page out the way print will: print media, paper width.
            page.set_viewport_size({"width": CONTENT_WIDTH_PX, "height": int(CONTENT_HEIGHT_PX)})
            page.emulate_media(media="print")
            page.set_content(html, wait_until="load")
            # Body's laid-out box, not scrollHeight: scrollHeight is floored
            # at the viewport height, so a page that fits would always read
            # as "exactly full". print.css zeroes body margins, so this is
            # the content height.
            content_height = page.evaluate("document.body.getBoundingClientRect().height")
            # No format/margin args on purpose: prefer_css_page_size hands
            # geometry control to @page in print.css — one source of truth.
            page.pdf(
                path=str(out),
                prefer_css_page_size=True,
                print_background=True,
            )
        finally:
            page.close()

        return RenderResult(
            pdf_path=out,
            html_path=html_path,
            page_count=len(PdfReader(str(out)).pages),
            fill=content_height / CONTENT_HEIGHT_PX,
            scale=scale,
        )


def render_resume(
    data: dict | Resume,
    scale: float = 1.0,
    out_path: Path | str | None = None,
) -> tuple[Path, int]:
    """One-shot render (the roadmap signature): data -> (pdf_path, page_count)."""
    with Renderer() as renderer:
        result = renderer.render(data, scale=scale, out_path=out_path)
    return result.pdf_path, result.page_count


if __name__ == "__main__":
    # python -m src.tools.render
    with Renderer() as r:
        res = r.render(Resume.load(ROOT / "assets" / "resume.json"))
    print(f"{res.pdf_path} — {res.page_count} page(s), fill {res.fill:.2f}")
