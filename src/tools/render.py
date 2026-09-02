"""Deterministic render tool: resume data -> validated HTML -> PDF -> page count.

Roadmap invariant #2 lives here: the page count is *measured* by pypdf
from the actual artifact, never estimated by a model. Downstream, the
fit loop and the Judge both read this number instead of asking an LLM
"does this look like two pages?".

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
# ~8.4pt — the readability floor. fit.py (Phase 3) imports these instead
# of inventing its own.
SCALE_MIN = 0.80
SCALE_MAX = 1.20

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
    page_count: int


class Renderer:
    """Reusable Chromium session.

    Browser startup (~1s) dominates render time, and the fit loop renders
    up to 3-4 times per job — so hold one browser open and reuse it:

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
            page.set_content(html, wait_until="load")
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
    # Phase 1 acceptance command:  python -m src.tools.render
    pdf_path, pages = render_resume(Resume.load(ROOT / "assets" / "resume.json"))
    print(f"{pdf_path} — {pages} page(s)")
