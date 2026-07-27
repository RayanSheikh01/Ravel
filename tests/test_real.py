from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from tools.real import make_real
from tools.sim import SimWorld


def _docker_up() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True,
                              timeout=15).returncode == 0
    except Exception:
        return False


requires_docker = pytest.mark.skipif(not _docker_up(), reason="docker unavailable")


def test_real_file_tools_hit_disk(tmp_path):
    """read/write/list touch real disk and mirror into world.files. No container."""
    world = SimWorld(files={"seed.txt": "hi\n"})
    reg = make_real(world, str(tmp_path))

    # make_real seeds the sandbox from world.files
    assert (tmp_path / "seed.txt").read_text() == "hi\n"

    # write hits real disk and mirrors into world.files (for goal_check)
    reg.dispatch("write_file", {"path": "out.txt", "content": "abc"})
    assert (tmp_path / "out.txt").read_text() == "abc"
    assert world.files["out.txt"] == "abc"

    # read comes off disk
    r = reg.dispatch("read_file", {"path": "out.txt"})
    assert r.ok and r.content == "abc"

    # list walks real files
    r = reg.dispatch("list_dir", {"prefix": ""})
    assert {"out.txt", "seed.txt"} <= set(r.content.split())

    # path escape is refused (dispatch turns the raise into ok=False)
    r = reg.dispatch("read_file", {"path": "../escape.txt"})
    assert not r.ok


@requires_docker
def test_run_executes_in_container(tmp_path):
    reg = make_real(SimWorld(), str(tmp_path))
    r = reg.dispatch("run", {"cmd": "python -c 'print(6*7)'"})
    assert r.ok and "42" in r.content


@requires_docker
def test_sandbox_round_trip(tmp_path):
    """A file the container writes under /work is visible host-side and to read_file."""
    world = SimWorld()
    reg = make_real(world, str(tmp_path))
    r = reg.dispatch("run", {"cmd": "echo hello > made.txt"})
    assert r.ok
    assert (tmp_path / "made.txt").read_text().strip() == "hello"
    r = reg.dispatch("read_file", {"path": "made.txt"})
    assert r.ok and "hello" in r.content


@requires_docker
def test_container_isolation(tmp_path):
    reg = make_real(SimWorld(), str(tmp_path))
    # no network: any outbound request fails
    r = reg.dispatch("run", {"cmd":
        "python -c \"import urllib.request as u; u.urlopen('http://example.com', timeout=5)\""})
    assert not r.ok
    # read-only root fs: writing outside the /work mount fails
    r = reg.dispatch("run", {"cmd": "echo x > /evil"})
    assert not r.ok
    # the host filesystem is not mounted; only /work exists to write into
    r = reg.dispatch("run", {"cmd": "test -w /work && echo ok"})
    assert r.ok and "ok" in r.content


@requires_docker
def test_run_timeout(tmp_path, monkeypatch):
    import tools.real as real
    monkeypatch.setattr(real, "RUN_TIMEOUT", 3)
    reg = make_real(SimWorld(), str(tmp_path))
    r = reg.dispatch("run", {"cmd": "sleep 30"})
    assert not r.ok and "timed out" in r.content


@requires_docker
def test_output_capped(tmp_path, monkeypatch):
    import tools.real as real
    monkeypatch.setattr(real, "MAX_OUTPUT", 100)
    reg = make_real(SimWorld(), str(tmp_path))
    r = reg.dispatch("run", {"cmd": "python -c \"print('x'*10000)\""})
    assert len(r.content) <= 100


@requires_docker
@pytest.mark.skipif(os.getenv("RUN_OLLAMA") != "1", reason="live model; set RUN_OLLAMA=1")
def test_fix_and_run_on_real_backend(tmp_path):
    from agent import run_agent
    from tasks import REGISTRY, load_tasks

    load_tasks("tasks")  # register the migrated .yaml tasks
    task = REGISTRY["fix_and_run"]
    world = task.setup()
    reg = task.make_registry(world, backend="real", sandbox_dir=str(tmp_path))
    run_agent(task.prompt, reg, task_id=task.id, backend="real")
    assert task.goal_check(world) == 1.0
