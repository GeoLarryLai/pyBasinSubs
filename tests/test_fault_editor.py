# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
import numpy as np
import pytest

from pybasinsubs import faults as F, flex2d
from pybasinsubs.fault_tracer import FaultStore, FaultTracer, vg1_example


@pytest.fixture(scope="module")
def vg1():
    return vg1_example()


def test_user_file_roundtrip(tmp_path):
    st = {"A": np.array([[1000.0, 2000.0], [1500.0, 3000.0], [2500.0, 5000.0]]), "B 2": np.array([[9e3, 1e3], [1e4, 4e3]])}
    p = F.write_user_faults(st, tmp_path / "u.csv", {"A": 6, "B 2": None})
    s, t = F.read_user_faults(p)
    assert list(s) == ["A", "B 2"] and t == {"A": 6, "B 2": None}
    for k in st:
        assert np.allclose(s[k], st[k])
    assert list(__import__("pandas").read_csv(p).columns[:4]) == ["fault_id", "x", "z", "order"]


def test_store_edit_undo_reset_save(vg1, tmp_path):
    sec, tops, base = vg1
    s = FaultStore(sec.x, tops, base, path=tmp_path / "f.csv")
    s.reset("curved_midramp"); assert len(s.sticks) == 5
    s.reset("straight"); assert len(s.sticks) == 5
    s.reset("curved_crest"); crest = s._snap()[0]
    fid = s.add([[5e4, 1e3], [5.2e4, 4e3]]); assert fid in s.sticks and len(s.sticks) == 6
    j = s.insert_node(fid); assert len(s.sticks[fid]) == 3 and j == 1
    s.move_node(fid, 1, (5.15e4, 2.6e3)); assert np.allclose(s.sticks[fid][1], (5.15e4, 2.6e3))
    assert s.delete_node(fid, 1) and len(s.sticks[fid]) == 2 and not s.delete_node(fid, 0)
    assert s.rename(fid, "NEW") and "NEW" in s.sticks and fid not in s.sticks
    s.set_tip("NEW", 5); s.save()
    assert s.autosave_path.exists()
    s.delete("NEW"); s.reset("empty"); assert s.sticks == {}
    s.undo(); s.undo(); assert "NEW" in s.sticks
    s.redo(); assert "NEW" not in s.sticks
    s.reset("saved"); assert "NEW" in s.sticks and s.tips["NEW"] == 5 and len(s.sticks) == 6
    for k in crest:
        assert np.allclose(s.sticks[k], crest[k])
    fs = s.faults(); assert len(fs) == 6 and all(f.curve is not None for f in fs)


def test_faults_source(vg1, tmp_path):
    sec, tops, base = vg1
    fa, src = F.load_faults("auto", sec.x, tops, base)
    fm, _ = F.load_faults("curved_midramp", sec.x, tops, base)
    assert src == "auto" and np.allclose([f.x0 for f in fa], [f.x0 for f in fm])
    fc, _ = F.load_faults("curved_crest", sec.x, tops, base)
    fs, _ = F.load_faults("straight", sec.x, tops, base)
    assert all(f.curve is None for f in fs) and fc[0].x0 < fm[0].x0
    p = F.write_user_faults(F.fault_polylines(fm), tmp_path / "u.csv", None)
    fu, used = F.load_faults(p, sec.x, tops, base)
    assert used == str(p) and np.allclose([f.x0 for f in fu], [f.x0 for f in fm], atol=500)
    r = flex2d.restore(sec, str(p), method="cantilever", Te=1500.0, beta=1.15)
    assert np.isfinite(flex2d.restoration_metrics(r, len(r["ages"]) - 1, sec.x, fu)["std"])
    if not F.user_faults_path().exists():
        assert F.load_faults("user", sec.x, tops, base)[1] == "auto"
        with pytest.raises(FileNotFoundError):
            F.load_faults("user", sec.x, tops, base, fallback=False)


def test_tracer_starts_and_panel(vg1, tmp_path):
    sec, tops, base = vg1
    tr = FaultTracer(sec, tops, base=base, path=tmp_path / "x.csv")
    assert len(tr.sticks()) == 5 and not tr.store._undo
    tr.save(); tr2 = FaultTracer(sec, tops, base=base, path=tmp_path / "x.csv")
    assert set(tr2.sticks()) == set(tr.sticks())
    pytest.importorskip("panel"); pytest.importorskip("bokeh")
    app = tr2.panel()
    assert app is not None and len(tr2.src.data["xs"]) == 5
    d = dict(tr2.src.data); d["xs"] = d["xs"] + [[10.0, 11.0]]; d["ys"] = d["ys"] + [[1000.0, 3000.0]]
    d["fault_id"] = d["fault_id"] + [""]; d["color"] = d["color"] + [""]
    tr2.src.data = d                                       # as PolyDrawTool would
    assert len(tr2.store.sticks) == 6 and "" not in tr2.store.sticks
    tr2.store.undo(); assert len(tr2.store.sticks) == 5


def test_json_roundtrip_and_restore(vg1, tmp_path):
    import json
    sec, tops, base = vg1
    s = FaultStore(sec.x, tops, base, path=tmp_path / "f.json")
    s.reset("curved_midramp"); s.add([[5e4, 1e3], [5.1e4, 2.5e3], [5.3e4, 4e3]], fid="mine", tip=5)
    s.vertical_exaggeration = 3.5
    p = s.save(); assert p.suffix == ".json" and s.autosave_path.name == "f.autosave.json" and s.autosave_path.exists()
    doc = json.loads(p.read_text())
    assert doc["version"] == 1 and doc["units"] == "m" and doc["section_id"] == "VG1" and doc["vertical_exaggeration"] == 3.5
    assert set(doc["metadata"]) == {"created", "modified", "pybasinsubs_version"}
    f = {d["id"]: d for d in doc["faults"]}
    assert f["mine"]["tip_unit"] == 5 and f["mine"]["method_origin"] == "user" and f["mine"]["nodes"][1] == [5.1e4, 2.5e3]
    assert f["auto-1"]["method_origin"] == "curved_midramp" and f["auto-1"]["name"] == "auto-1"
    created = doc["metadata"]["created"]; s.save(); assert json.loads(p.read_text())["metadata"]["created"] == created
    s2 = FaultStore(sec.x, tops, base, path=tmp_path / "f.json"); s2.load()
    assert list(s2.sticks) == list(s.sticks) and s2.tips == s.tips and s2.origins == s.origins
    assert s2.vertical_exaggeration == 3.5
    for k in s.sticks:
        assert np.allclose(s2.sticks[k], s.sticks[k])
    pc = s.save(fmt="csv"); assert pc.suffix == ".csv"
    s3 = FaultStore(sec.x, tops, base, path=tmp_path / "f.json"); s3.load(fmt="csv"); assert s3.tips["mine"] == 5
    sk, tp = F.read_user_faults(p); assert tp["mine"] == 5 and np.allclose(sk["mine"], s.sticks["mine"])
    fu, used = F.load_faults(p, sec.x, tops, base); assert used == str(p) and len(fu) == 6
    r = flex2d.restore(sec, p, method="domino")
    assert np.isfinite(flex2d.restoration_metrics(r, len(r["ages"]) - 1, sec.x, fu)["std"])
    gj = F.read_fault_sticks(F.user_faults_path("vg1_fault_sticks.geojson"))     # GeoJSON still read as sticks
    assert len(gj) >= 2


def test_user_prefers_json(vg1, tmp_path, monkeypatch):
    sec, tops, base = vg1
    monkeypatch.setattr(F, "user_faults_path", lambda name=F.USER_FAULTS_FILE: tmp_path / name)
    assert F.stored_user_faults() is None
    st = {"A": np.array([[1e4, 2e3], [1.2e4, 5e3]])}
    F.write_user_faults(st, tmp_path / F.USER_FAULTS_CSV)
    assert F.stored_user_faults().suffix == ".csv"
    F.write_user_faults({"J": st["A"]}, tmp_path / F.USER_FAULTS_FILE)
    assert F.stored_user_faults().suffix == ".json"
    fs, used = F.load_faults("user", sec.x, tops, base)
    assert used.endswith(".json") and [f.label for f in fs] == ["J"]
