from tools.sim import SimWorld, make_sim


def test_write_read_round_trip():
    reg = make_sim(SimWorld())
    assert reg.dispatch("write_file", {"path": "a.txt", "content": "hi"}).ok
    res = reg.dispatch("read_file", {"path": "a.txt"})
    assert res.ok is True
    assert res.content == "hi"


def test_missing_file_not_ok():
    reg = make_sim(SimWorld())
    assert reg.dispatch("read_file", {"path": "nope.txt"}).ok is False


def test_vanish_makes_read_fail():
    # The hook the `vanish` failure mode exploits later.
    world = SimWorld()
    reg = make_sim(world)
    reg.dispatch("write_file", {"path": "b.txt", "content": "x"})
    assert reg.dispatch("read_file", {"path": "b.txt"}).ok is True
    del world.files["b.txt"]  # resource disappears mid-run
    assert reg.dispatch("read_file", {"path": "b.txt"}).ok is False
