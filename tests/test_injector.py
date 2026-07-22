from injector import FailureRule, InjectingRegistry
from tools.sim import SimWorld, make_sim


def test_step_trigger_fires_on_third_dispatch_once():
    world = SimWorld(files={"a.txt": "hi"})
    reg = InjectingRegistry(make_sim(world), FailureRule("hard_error", "step:2"))
    results = [reg.dispatch("read_file", {"path": "a.txt"}) for _ in range(4)]
    injected = [r.meta.get("injected", False) for r in results]
    assert injected == [False, False, True, False]  # 3rd dispatch only
    assert reg.injection_step == 2


def test_first_call_fires_on_first_matching_call():
    world = SimWorld(files={"a.txt": "hi"})
    reg = InjectingRegistry(make_sim(world), FailureRule("vanish", "first_call:read_file"))
    reg.dispatch("write_file", {"path": "x", "content": "y"})  # not read_file
    first = reg.dispatch("read_file", {"path": "a.txt"})
    second = reg.dispatch("read_file", {"path": "a.txt"})
    assert first.meta.get("injected") is True and first.ok is False
    assert second.meta.get("injected", False) is False and second.ok is True


def test_touch_trigger_and_determinism():
    def run():
        world = SimWorld(files={"data/b.csv": "1"})
        reg = InjectingRegistry(make_sim(world), FailureRule("vanish", "touch:data/b.csv"))
        reg.dispatch("read_file", {"path": "data/a.csv"})  # no match
        hit = reg.dispatch("read_file", {"path": "data/b.csv"})
        return hit.meta.get("injected", False), reg.injection_step

    assert run() == (True, 1)
    assert run() == run()  # identical across runs
