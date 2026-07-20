"""Render per-case Open Graph images (1200×630) into ui/og/case/.

Build-time tool — requires playwright + a local Chrome:
    pip install playwright   # then it uses channel="chrome"

Each card: site-dark background, case title, status pill, date/location,
summary, record stats, and a strip of record thumbnails. Re-run after
build_case_pages.py whenever cases change.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "ui"
OUT = UI / "og" / "case"
W, H = 1200, 630

BG = "#0c1014"
FG = "#e6eef2"
DIM = "#8696a0"
SIGNAL = "#7fd4b5"
AMBER = "#e4b463"
RULE = "#2f3d49"
PANEL = "#161e25"


def esc(s: str) -> str:
    return html.escape(s or "", quote=True)


def thumb_of(d: dict) -> str:
    t = d.get("thumb_small") or ""
    if not t:
        tl = d.get("thumbnail_local")
        if isinstance(tl, list) and tl:
            t = tl[0]
        elif isinstance(tl, str):
            t = tl
    return t


STATUS_COLORS = {
    "unresolved": (SIGNAL, "rgba(127,212,181,0.12)"),
    "disputed": (AMBER, "rgba(228,180,99,0.12)"),
    "debunked": (DIM, "rgba(134,150,160,0.10)"),
}


def page_html(*, kicker: str, title: str, status: str, chips: list[str],
              summary: str, stats: str, thumbs: list[str]) -> str:
    fg, bg = STATUS_COLORS.get(status, STATUS_COLORS["unresolved"])
    chips_html = "".join(
        f'<span style="font:500 20px \'JetBrains Mono\',monospace;color:{DIM};'
        f'border:1px solid {RULE};border-radius:6px;padding:5px 14px;background:{PANEL}">{esc(c)}</span>'
        for c in chips if c
    )
    thumbs_html = "".join(
        f'<img src="/{esc(t)}" style="width:172px;height:128px;object-fit:cover;'
        f'border-radius:8px;border:1px solid {RULE};background:{PANEL}"/>'
        for t in thumbs
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"/>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  * {{ margin:0; box-sizing:border-box; }}
  body {{
    width:{W}px; height:{H}px; background:{BG}; color:{FG};
    font-family:'Geist',sans-serif; position:relative; overflow:hidden;
    background-image:
      radial-gradient(900px 500px at 85% 20%, rgba(127,212,181,0.07), transparent 60%),
      linear-gradient(rgba(123,184,212,0.045) 1px, transparent 1px),
      linear-gradient(90deg, rgba(123,184,212,0.045) 1px, transparent 1px);
    background-size:auto, 80px 80px, 80px 80px;
  }}
  .glow {{
    position:absolute; right:-160px; top:-160px; width:520px; height:520px;
    border-radius:50%; border:1px solid rgba(127,212,181,0.25);
    box-shadow:0 0 120px 30px rgba(127,212,181,0.08), inset 0 0 80px rgba(127,212,181,0.06);
  }}
</style></head>
<body>
  <div class="glow"></div>
  <div style="position:absolute;left:64px;top:64px;right:420px;bottom:64px;display:flex;flex-direction:column">
    <div style="font:600 19px 'JetBrains Mono',monospace;letter-spacing:0.22em;color:{SIGNAL}">{esc(kicker)}</div>
    <div style="font:700 56px 'Geist',sans-serif;line-height:1.08;margin:18px 0 20px;letter-spacing:-0.01em">{esc(title)}</div>
    <div style="display:flex;gap:10px;align-items:center;margin-bottom:20px">
      <span style="font:600 18px 'JetBrains Mono',monospace;letter-spacing:0.1em;color:{fg};background:{bg};border:1px solid {fg};border-radius:6px;padding:5px 14px">{esc(status.upper())}</span>
      {chips_html}
    </div>
    <div style="font:400 23px 'Geist',sans-serif;line-height:1.45;color:{DIM};max-width:34em">{esc(summary)}</div>
    <div style="flex:1"></div>
    <div style="font:500 21px 'JetBrains Mono',monospace;color:{SIGNAL};letter-spacing:0.06em">{esc(stats)}</div>
    <div style="font:400 18px 'JetBrains Mono',monospace;color:{DIM};margin-top:14px">UFO/UAP Disclosure Archive · uapdisclosuremirror.com</div>
  </div>
  <div style="position:absolute;right:56px;top:50%;transform:translateY(-50%);display:grid;grid-template-columns:repeat(2,172px);gap:14px">
    {thumbs_html}
  </div>
</body></html>"""


def main() -> None:
    from playwright.sync_api import sync_playwright

    # Serve the repo root so record thumbnails (/raw/...) resolve in-page.
    import functools
    import http.server
    import socketserver
    import threading

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{port}"

    cases = json.loads((UI / "cases.json").read_text())
    lite = {d["id"]: d for d in json.loads((UI / "records-lite.json").read_text())}
    OUT.mkdir(parents=True, exist_ok=True)

    jobs = []
    for cid, c in cases.items():
        ids = (c.get("records") or []) + (c.get("era") or [])
        thumbs = []
        for rid in ids:
            t = thumb_of(lite.get(rid, {}))
            if t and t not in thumbs and (ROOT / t).is_file():
                thumbs.append(t)
            if len(thumbs) == 6:
                break
        n = c.get("count", 0)
        jobs.append({
            "file": OUT / f"{cid}.png",
            "html": page_html(
                kicker=f"CASE DOSSIER · {c.get('kind', 'incident').upper()}",
                title=c.get("name", cid),
                status=(c.get("status") or "unresolved").lower(),
                chips=[c.get("date", ""), c.get("location", "")],
                summary=c.get("summary", ""),
                stats=f"{n} declassified record{'s' if n != 1 else ''} in the archive",
                thumbs=thumbs,
            ),
        })

    jobs.append({
        "file": OUT / "index.png",
        "html": page_html(
            kicker="CASE DOSSIERS",
            title="Every major UAP incident, dossier by dossier",
            status="unresolved",
            chips=[f"{len(cases)} cases"],
            summary="Roswell, Nimitz, Rendlesham, Blue Book and more — each dossier pulls every declassified record that references it.",
            stats=f"{sum(c.get('count', 0) for c in cases.values())} record references across the archive",
            thumbs=[],
        ),
    })

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=2)
        for job in jobs:
            page.goto(origin + "/ui/index.html")
            page.set_content(job["html"].replace('src="/', f'src="{origin}/'), wait_until="networkidle")
            page.wait_for_function("document.fonts.status === 'loaded'")
            page.wait_for_timeout(150)
            page.screenshot(path=str(job["file"]), clip={"x": 0, "y": 0, "width": W, "height": H})
            print("→", job["file"].relative_to(ROOT))
        browser.close()
    httpd.shutdown()


if __name__ == "__main__":
    main()
