import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest


def test_tasks_route():
    from server import Handler

    srv = ThreadingHTTPServer(("localhost", 0), Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        conn = http.client.HTTPConnection("localhost", port)
        conn.request("GET", "/tasks")
        resp = conn.getresponse()
        assert resp.status == 200
        tasks = json.loads(resp.read())
        ids = {t["id"] for t in tasks}
        assert {"sum_csvs", "fix_and_run", "fetch_merge"} <= ids
    finally:
        srv.shutdown()


def test_trajectory_serialization():
    from run import run_tasks
    from collections import Counter
    import json

    rows, by_task, by_mode = run_tasks(
        task_ids=["fix_and_run"],
        seeds=0,
        model="qwen2.5-coder",
        backend="sim"
    )

    # Check that the trajectory key exists and is serializable
    for row in rows:
        assert "trajectory" in row
        # Ensure it can be serialized to JSON
        json_str = json.dumps(row["trajectory"])
        # Ensure it can be deserialized back to a list of dicts
        traj_list = json.loads(json_str)
        assert isinstance(traj_list, list)
        for t in traj_list:
            assert isinstance(t, dict)