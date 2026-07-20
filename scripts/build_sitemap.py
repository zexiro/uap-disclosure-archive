"""Build sitemap.xml + robots.txt for the archive.

Covers: home, the core app pages, the case dossier index, every case
dossier, and every record page (record.html?id=…). Re-run whenever the
archive index or case set changes (safe to chain after
build_case_pages.py).
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "ui"
SITE = "https://uapdisclosuremirror.com"

CORE_PAGES = [
    ("/ui/", "1.0", "daily"),
    ("/ui/search.html", "0.6", "weekly"),
    ("/ui/map.html", "0.6", "weekly"),
    ("/ui/timeline.html", "0.6", "weekly"),
    ("/ui/graph.html", "0.6", "weekly"),
    ("/ui/ask.html", "0.5", "weekly"),
    ("/ui/case/", "0.8", "weekly"),
]


def url_entry(loc: str, lastmod: str, priority: str, changefreq: str) -> str:
    return (
        "  <url>\n"
        f"    <loc>{escape(loc)}</loc>\n"
        f"    <lastmod>{lastmod}</lastmod>\n"
        f"    <changefreq>{changefreq}</changefreq>\n"
        f"    <priority>{priority}</priority>\n"
        "  </url>\n"
    )


def main() -> None:
    today = date.today().isoformat()
    cases = json.loads((UI / "cases.json").read_text())
    docs = json.loads((UI / "records-lite.json").read_text())

    out = [
        '<?xml version="1.0" encoding="UTF-8"?>\n',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n',
    ]
    for path, pri, freq in CORE_PAGES:
        out.append(url_entry(SITE + path, today, pri, freq))
    for cid in sorted(cases):
        out.append(url_entry(f"{SITE}/ui/case/{cid}/", today, "0.7", "monthly"))
    for d in docs:
        out.append(url_entry(f"{SITE}/ui/record.html?id={d['id']}", today, "0.5", "monthly"))
    out.append("</urlset>\n")

    (ROOT / "sitemap.xml").write_text("".join(out))
    (ROOT / "robots.txt").write_text(
        "User-agent: *\n"
        "Allow: /\n"
        "Disallow: /api/\n"
        "\n"
        f"Sitemap: {SITE}/sitemap.xml\n"
    )
    print(f"sitemap.xml: {len(cases) + len(docs) + len(CORE_PAGES)} urls · robots.txt")


if __name__ == "__main__":
    main()
