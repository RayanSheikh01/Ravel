import pytest


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