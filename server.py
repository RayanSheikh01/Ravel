"""Stdlib dashboard server for the plan-repair harness.

Serves the single-page UI, the current results.json, the task list, and a
blocking POST /run that reuses run.py's run_tasks. No deps beyond stdlib.
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from run import run_tasks  # noqa: F401 — importing run.py runs load_tasks(), populating REGISTRY
from tasks import REGISTRY, add_task_from_yaml

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
                {"id": t.id, "mode": t.default_rule.mode if t.default_rule else "none",
                 "uploaded": t.uploaded}
                for t in REGISTRY.values()
            ])
        else:
            self._json({"error": "not found"}, 404)

    def _read_body(self) -> bytes:
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def do_POST(self):
        if self.path == "/upload":
            self._upload()
        elif self.path == "/run":
            self._run()
        else:
            self._json({"error": "not found"}, 404)

    def _upload(self):
        try:
            task = add_task_from_yaml(self._read_body().decode("utf-8"))
        except Exception as e:  # validation failure -> 400, message names the reason
            self._json({"error": str(e)}, 400)
            return
        self._json({"id": task.id,
                    "mode": task.default_rule.mode if task.default_rule else "none",
                    "uploaded": True})

    def _run(self):
        body = json.loads(self._read_body() or b"{}")
        backend = body.get("backend", "sim")
        # Step 5 gate: uploaded tasks never touch the real backend.
        if backend == "real":
            blocked = [t for t in body.get("tasks", [])
                       if getattr(REGISTRY.get(t), "uploaded", False)]
            if blocked:
                self._json({"error": f"uploaded tasks are sim-only: {blocked}"}, 403)
                return
        try:
            # ponytail: single blocking run, no job queue -- add streaming only if latency hurts.
            rows, by_task, by_mode = run_tasks(
                task_ids=body["tasks"],
                seeds=body.get("seeds", 3),
                model=body.get("model", "qwen2.5-coder"),
                backend=backend,
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
