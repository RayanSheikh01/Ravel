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


def test_upload_and_sim_only_gate():
    from server import Handler

    srv = ThreadingHTTPServer(("localhost", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    conn = http.client.HTTPConnection("localhost", port)
    good = "id: srv_up\nprompt: p\ngoal: {check: file_exists, file: o}\n"
    try:
        # upload validates + registers
        conn.request("POST", "/upload", body=good)
        resp = conn.getresponse()
        assert resp.status == 200
        assert json.loads(resp.read())["id"] == "srv_up"

        # shows up in /tasks flagged uploaded
        conn.request("GET", "/tasks")
        tasks = {t["id"]: t for t in json.loads(conn.getresponse().read())}
        assert tasks["srv_up"]["uploaded"] is True

        # real backend refused for an uploaded task (403, before any run)
        conn.request("POST", "/run",
                     body=json.dumps({"tasks": ["srv_up"], "backend": "real", "seeds": 0}))
        assert conn.getresponse().status == 403

        # malformed task -> 400
        conn.request("POST", "/upload", body="id: x\n")  # missing prompt + goal
        assert conn.getresponse().status == 400
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