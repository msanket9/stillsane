"""Per-run sample capture: what `check` actually saw, kept a while."""

from __future__ import annotations

import contextlib
import time

from conftest import sample

import stillsane.store.runs as runs_module
from stillsane.store.runs import RunSampleStore


def test_append_then_load_round_trips(tmp_path):
    store = RunSampleStore(tmp_path)
    store.append("run1", [sample("first"), sample("second")], store_raw=False)

    loaded = store.load("run1")
    assert [s.text for s in loaded] == ["first", "second"]


def test_loading_an_unknown_run_returns_none(tmp_path):
    assert RunSampleStore(tmp_path).load("nope") is None


def test_appending_no_samples_writes_nothing(tmp_path):
    store = RunSampleStore(tmp_path)
    store.append("run1", [], store_raw=False)
    assert store.load("run1") is None
    assert not store.root.exists()


def test_multiple_probes_in_one_run_share_a_file(tmp_path):
    """One `check` run can cover several probes; `check` calls `append` once
    per probe, all under the same `run_id`, so they must accumulate rather
    than overwrite each other.
    """
    store = RunSampleStore(tmp_path)
    store.append("run1", [sample("probe a's answer", probe="a")], store_raw=False)
    store.append("run1", [sample("probe b's answer", probe="b")], store_raw=False)

    loaded = store.load("run1")
    assert {s.probe_id for s in loaded} == {"a", "b"}


def test_a_crash_mid_append_does_not_corrupt_earlier_samples(tmp_path, monkeypatch):
    """A plain `open("a")` append has no way to undo a partial last line: a
    crash mid-write of the *second* probe's samples used to leave a truncated,
    unparseable line in the file, and `load` reads the whole file at once --
    so `json.loads` raising on that one bad line lost every sample recorded
    for the *first* probe too, in the same run.
    """
    store = RunSampleStore(tmp_path)
    store.append("run1", [sample("first probe's answer, must survive")], store_raw=False)

    def crashes_mid_write(path, content):
        path.with_name(f"{path.name}.orphan.tmp").write_text(content[: len(content) // 2])
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr(runs_module, "atomic_write", crashes_mid_write)
    with contextlib.suppress(RuntimeError):
        store.append("run1", [sample("second probe's answer, lost to the crash")], store_raw=False)

    loaded = store.load("run1")
    assert loaded is not None
    assert [s.text for s in loaded] == ["first probe's answer, must survive"]


def test_raw_is_stripped_by_default(tmp_path):
    """Same secrets consideration as baseline capture (`TargetConfig.store_raw`):
    nothing reads `Sample.raw` back, and it can carry tenant data for a
    `type: http` target against a user's own app.
    """
    store = RunSampleStore(tmp_path)
    s = sample("hello")
    s.raw = {"secret_tenant_field": "do-not-leak"}
    store.append("run1", [s], store_raw=False)

    loaded = store.load("run1")
    assert loaded[0].raw == {}


def test_raw_is_kept_when_the_target_opted_in(tmp_path):
    store = RunSampleStore(tmp_path)
    s = sample("hello")
    s.raw = {"secret_tenant_field": "not actually a secret here"}
    store.append("run1", [s], store_raw=True)

    loaded = store.load("run1")
    assert loaded[0].raw == {"secret_tenant_field": "not actually a secret here"}


def test_prune_keeps_only_the_most_recently_written_runs(tmp_path):
    """A `watch` loop firing every few minutes must not grow this directory
    without bound.
    """
    store = RunSampleStore(tmp_path, keep=3)
    for i in range(6):
        store.append(f"run{i}", [sample(f"text {i}")], store_raw=False)
        store.prune()
        time.sleep(0.01)  # distinct mtimes to order by

    remaining = {d.name for d in store.root.iterdir()}
    assert remaining == {"run3", "run4", "run5"}
    assert store.load("run0") is None
    assert store.load("run5") is not None


def test_prune_is_a_noop_before_anything_is_written(tmp_path):
    RunSampleStore(tmp_path).prune()  # must not raise
