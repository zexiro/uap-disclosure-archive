#!/usr/bin/env python3
"""Tiny static file server tuned for this project.

- Serves the entire repo root so /ui/, /raw/, /vault/ all resolve.
- Redirects "/" → "/ui/" so the search UI is the home page.
- Sets long cache headers on raw/* (immutable assets) and short on ui/* (changes often).
- Honours the PORT env var (Railway).
"""
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("", "/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/ui/")
            self.end_headers()
            return
        return super().do_GET()

    def end_headers(self):
        # Range support is automatic in SimpleHTTPRequestHandler since 3.7
        if self.path.startswith("/raw/") and any(
            self.path.endswith(ext) for ext in (".pdf", ".mp4", ".jpg", ".jpeg", ".png", ".webm")
        ):
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        elif self.path.startswith("/ui/"):
            self.send_header("Cache-Control", "public, max-age=300")
        # CORS so iframes / clients work cleanly
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stdout.write("[http] %s %s\n" % (self.log_date_time_string(), fmt % args))
        sys.stdout.flush()


def main():
    port = int(os.environ.get("PORT", "8000"))
    addr = os.environ.get("BIND", "0.0.0.0")
    httpd = HTTPServer((addr, port), Handler)
    print(f"[http] serving on http://{addr}:{port}/", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
