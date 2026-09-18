"""
server.py
---------
A tiny local API + static file server built entirely on Python's standard
library (http.server) -- no Flask, no FastAPI, no third-party dependency
of any kind. It serves the frontend files and exposes a handful of JSON
endpoints the frontend's app.js talks to.

Routes:
    GET  /                -> frontend/index.html
    GET  /app.js           -> frontend/app.js
    GET  /style.css        -> frontend/style.css
    POST /api/chat         -> {message} -> full pipeline result
    GET  /api/context      -> current mood/case-file snapshot
    GET  /api/history      -> recent chat log (for reloading the page)
    POST /api/mood         -> {mood} -> force-set the mood
    POST /api/reset        -> wipe the case file and start over
"""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import database, pipeline

FRONTEND_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend"
)

STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}


class SideEyeHandler(BaseHTTPRequestHandler):
    server_version = "SideEye/1.0"

    def log_message(self, fmt, *args):
        # Quieter console output than the default, still shows requests.
        print(f"[sideeye] {self.address_string()} - {fmt % args}")

    # ---------------------------------------------------------- helpers --

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, filename: str, content_type: str):
        path = os.path.join(FRONTEND_DIR, filename)
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError:
            self._send_json({"error": "not found"}, status=404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    # ------------------------------------------------------------- GET --

    def do_GET(self):
        if self.path in STATIC_FILES:
            filename, content_type = STATIC_FILES[self.path]
            self._send_static(filename, content_type)
            return

        if self.path == "/api/context":
            self._send_json(pipeline.get_snapshot())
            return

        if self.path.startswith("/api/history"):
            self._send_json({"history": pipeline.get_history()})
            return

        self._send_json({"error": "not found"}, status=404)

    # ------------------------------------------------------------ POST --

    def do_POST(self):
        if self.path == "/api/chat":
            data = self._read_json_body()
            message = (data.get("message") or "").strip()
            if not message:
                self._send_json({"error": "message is required"}, status=400)
                return
            try:
                result = pipeline.process_message(message)
                self._send_json(result)
            except Exception as exc:  # noqa: BLE001 - surface any bug as JSON, don't crash the server
                self._send_json({"error": f"internal error: {exc}"}, status=500)
            return

        if self.path == "/api/mood":
            data = self._read_json_body()
            mood = (data.get("mood") or "").strip()
            self._send_json(pipeline.set_mood(mood))
            return

        if self.path == "/api/reset":
            self._send_json(pipeline.reset_session())
            return

        self._send_json({"error": "not found"}, status=404)


def run(port: int = 8765):
    database.init_db()
    server = ThreadingHTTPServer(("127.0.0.1", port), SideEyeHandler)
    print(f"SideEye is listening on http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    run()
