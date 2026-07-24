"""Stdlib dashboard server for the plan-repair harness.

Serves the single-page UI, the current results.json, the task list, and a
blocking POST /run that reuses run.py's run_tasks. No deps beyond stdlib.
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from run import run_tasks  # noqa: F401 — importing run.py runs load_tasks(), populating REGISTRY
from tasks import REGISTRY

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "web", "index.html")
RESULTS = os.path.join(HERE, "results.json")


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode())

    def do_GET(self):
        if self.path == "/":
            try:
                with open(INDEX, "rb") as f:
                    body = f.read()
            except FileNotFoundError:
                self._json({"error": "web/index.html not built yet"}, 404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/results.json":
            try:
                with open(RESULTS, "rb") as f:
                    body = f.read()
            except FileNotFoundError:
                self._json([])  # missing -> empty, not 404
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/tasks":
            self._json([
                {"id": t.id, "mode": t.default_rule.mode if t.default_rule else "none"}
                for t in REGISTRY.values()
            ])
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/run":
            self._json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or "{}")
        try:
            # ponytail: single blocking run, no job queue -- add streaming only if latency hurts.
            rows, by_task, by_mode = run_tasks(
                task_ids=body["tasks"],
                seeds=body.get("seeds", 3),
                model=body.get("model", "qwen2.5-coder"),
                backend=body.get("backend", "sim"),
            )
        except Exception as e:  # Ollama-down etc. -- surface it, don't swallow.
            self._json({"error": str(e)}, 500)
            return
        self._json({"rows": rows, "by_task": by_task, "by_mode": by_mode})


def main():
    server = ThreadingHTTPServer(("localhost", 8000), Handler)
    print("serving on http://localhost:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
