#!/usr/bin/env python3
"""Tiny static file server tuned for this project.

- Serves the entire repo root so /ui/, /raw/, /vault/ all resolve.
- Redirects "/" → "/ui/" so the search UI is the home page.
- Sets long cache headers on raw/* (immutable assets) and short on ui/* (changes often).
- Honours the PORT env var (Railway).
"""
import os
import sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler


class Handler(SimpleHTTPRequestHandler):
    def _handle_special(self, write_body):
        # Liveness probe for Railway. Must stay cheap (no disk reads) so it
        # still answers when /raw/* image traffic is saturating worker threads.
        if self.path in ("/healthz", "/healthz/"):
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if write_body:
                self.wfile.write(body)
            return True
        if self.path in ("", "/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/ui/")
            self.end_headers()
            return True
        return False

    def do_GET(self):
        if self._handle_special(write_body=True):
            return
        return super().do_GET()

    def do_HEAD(self):
        if self._handle_special(write_body=False):
            return
        return super().do_HEAD()

    def end_headers(self):
        # Range support is automatic in SimpleHTTPRequestHandler since 3.7
        if self.path in ("/healthz", "/healthz/"):
            pass  # /healthz already wrote its own headers
        elif self.path.startswith("/raw/") and any(
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
    httpd = ThreadingHTTPServer((addr, port), Handler)
    httpd.daemon_threads = True
    print(f"[http] serving on http://{addr}:{port}/", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
