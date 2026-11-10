"""Browser app (pybasinsubs.webapp): routes for the 1-D column, the 2-D section and the fault editor, plus the launcher."""
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from pybasinsubs.webapp.routes import create_app  # noqa: E402


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(testing=True, faults_path=tmp_path / "faults.json"))


def test_index_options_and_templates(client):
    assert "pyBasinSubs" in client.get("/").text
    o = client.get("/api/options").json()
    assert o["defaults"]["n_mc"] == 50 and "cantilever" in o["methods"] and len(o["wells"]["viking_graben"]) > 1
    assert client.get("/api/template/horizons").text.splitlines()[0] == "well,x_m,unit,depth_m"
    assert client.get("/api/template/nope").status_code == 404
    for f in ("app.js", "faults.js", "app.css"):
        assert client.get("/static/" + f).status_code == 200


def test_column_route(client):
    r = client.post("/api/column", json=dict(source="North Viking Graben well (Sodir)", well="30/4-1", n=50, hf=60)).json()
    assert r["geohistory"] and r["maturity"] and r["summary"]["columns"] and len(r["summary_all"]["columns"]) >= len(r["summary"]["columns"])
    assert "30/4-1" in r["status"]


def test_section_route_and_errors(client):
    r = client.post("/api/section", json=dict(method="domino", n_mc=2)).json()
    assert "5 faults" in r["status"] and "method = domino" in r["status"] and r["figure"] and r["table"]["data"]
    for m in client.get("/api/options").json()["methods"]:                      # n_mc > 0 on every method
        assert client.post("/api/section", json=dict(method=m, n_mc=2)).status_code == 200
    bad = client.post("/api/section", json=dict(horizons="x_m,unit,depth_m\n0,a,1\n"))
    assert bad.status_code == 400 and "both" in bad.json()["detail"]
    bad = client.post("/api/section", json=dict(faults="a,b\n1,2\n"))
    assert bad.status_code == 400 and "fault_id, x, z" in bad.json()["detail"]


def test_fault_editor_add_undo_save_load(client, tmp_path):
    st = client.get("/api/faults/state").json()
    n0 = len(st["faults"]); assert n0 == 5 and st["extent"] and not st["can_undo"]
    assert client.get("/api/faults/image.png").content[:4] == b"\x89PNG"
    st = client.post("/api/faults/op", json=dict(op="add", nodes=[[1e4, 2e3], [1.2e4, 5e3]])).json()
    assert len(st["faults"]) == n0 + 1 and st["can_undo"]
    new = st["faults"][-1]["id"]
    st = client.post("/api/faults/op", json=dict(op="move_node", id=new, j=0, xz=[1.1e4, 2.5e3])).json()
    assert np.allclose(st["faults"][-1]["nodes"][0], [1.1e4, 2.5e3])
    assert client.post("/api/faults/op", json=dict(op="add", nodes=[[0, 0]])).status_code == 400
    for fmt in ("json", "csv"):
        client.post("/api/faults/save", json=dict(fmt=fmt))
        assert (tmp_path / f"faults.{fmt}").exists()
        assert client.get(f"/api/faults/download/{fmt}").status_code == 200
    st = client.post("/api/faults/op", json=dict(op="undo")).json()
    st = client.post("/api/faults/op", json=dict(op="undo")).json()
    assert len(st["faults"]) == n0
    st = client.post("/api/faults/load", json=dict(fmt="json")).json()
    assert len(st["faults"]) == n0 + 1                                          # the saved file has the added fault
    st = client.post("/api/faults/op", json=dict(op="reset", kind="empty")).json()
    assert st["faults"] == []
    r = client.get("/api/faults/auto/curved_midramp").json()
    assert len(r["faults"]) == 5


def test_fault_editor_restore_and_import(client):
    r = client.post("/api/faults/restore", json=dict(method="domino", Te=1500, Hd_km=8)).json()
    assert r["figure"] and "domino" in r["message"] and np.isfinite(r["metrics"]["std"])
    text = "fault_id,x,z\nA,10000,2000\nA,12000,5000\n"
    st = client.post("/api/faults/import", json=dict(name="f.csv", text=text)).json()
    assert [f["id"] for f in st["faults"]] == ["A"] and st["faults"][0]["origin"] == "imported"


def test_launcher_dry_run(monkeypatch, capsys):
    from pybasinsubs import webapp
    monkeypatch.setenv("PYBASINSUBS_DRY_RUN", "1")
    assert webapp.main([]) == 0 and webapp.tracer([]) == 0 and webapp.main(["--port", "9000", "--page", "section"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].endswith(":8767/") and out[1].endswith("#faults") and out[2] == "http://127.0.0.1:9000/#section"


def test_section_route_uses_advertised_mc_default(client, monkeypatch):
    """Omitting n_mc must agree with the value advertised to the browser."""
    from pybasinsubs.webapp import routes
    original = routes.core.run_section
    seen = []

    def quick_section(*args, **kwargs):
        seen.append(kwargs["n_mc"])
        kwargs["n_mc"] = 0  # this is a request-default test, not a Monte Carlo benchmark
        return original(*args, **kwargs)

    monkeypatch.setattr(routes.core, "run_section", quick_section)
    assert client.post("/api/section", json={"method": "domino"}).status_code == 200
    assert seen == [client.get("/api/options").json()["defaults"]["n_mc"]] == [50]


def test_explicit_growth_strip_does_not_run_on_page_load(client, monkeypatch):
    from pybasinsubs import growth_spatial
    original = growth_spatial.reconstruct_spatial_growth;calls = []

    def tracked(*a, **kw):
        calls.append(kw)
        return original(*a, **kw)

    monkeypatch.setattr(growth_spatial, "reconstruct_spatial_growth", tracked)
    page = client.get("/").text
    assert 'id="s_growth"' in page and 'id="s_growth_box" hidden' in page
    client.get("/api/options")
    assert not calls
    response = client.post("/api/section/growth", json={})
    assert response.status_code == 200, response.text
    data = response.json()
    assert len(calls) == 3 and "Fixed interpreted traces" in data["status"]  # primary + two sensitivity fits
    assert data["figure"] and len(data["history"]["data"]) == 50 and data["bins"]["data"]
    assert len(data["frames"]) == 11 and data["frames"][0]["age_Ma"] == 0.
    assert data["frames"][-1]["age_Ma"] == 252. and data["frames"][-1]["geometry_age_Ma"] == 170.3
    assert all(f["figure"] != f["figure_without_numbers"] for f in data["frames"])
    assert 'id="s_growth_time"' in page and 'aria-orientation="vertical"' in page and 'id="s_fault_numbers"' in page
    assert data["metadata"]["contact_auxiliary_translation_is_fault_slip"] is False
    assert not data["metadata"]["default_files_modified"]
    assert client.post("/api/section/growth", json={"horizons": "bad"}).status_code == 400


def test_ordinary_section_shows_only_present_day_no_second_gallery(client, monkeypatch):
    from pybasinsubs.webapp import routes
    from pybasinsubs import growth_spatial
    original = routes.core.run_section;seen = []

    def tracked(*args, **kwargs):
        seen.append(kwargs.get("ages"))
        return original(*args, **kwargs)

    def no_implicit_reconstruction(*args, **kwargs):
        raise AssertionError("Growth inference must not run during ordinary section / load")

    monkeypatch.setattr(routes.core, "run_section", tracked)
    monkeypatch.setattr(growth_spatial, "reconstruct_spatial_growth", no_implicit_reconstruction)
    assert client.get("/").status_code == 200
    assert client.get("/api/options").status_code == 200
    response = client.post("/api/section", json={"n_mc": 0})
    assert response.status_code == 200
    assert seen == [[0.]]


def test_home_original_logo_two_symbolic_cards_and_shared_footer(client):
    from html.parser import HTMLParser
    from pathlib import Path

    class HomeParser(HTMLParser):
        def __init__(self):
            super().__init__();self.home = False;self.cards = [];self.symbols = 0;self.logos = []
        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if attrs.get("id") == "page-home":self.home = True
            if tag == "img" and attrs.get("src") == "/static/logo.png":self.logos.append(attrs)
            if self.home and tag == "button":self.cards.append(attrs.get("data-go"))
            if self.home and tag == "svg":
                self.symbols += 1
                assert attrs["aria-hidden"] == "true" and attrs["viewbox"] == "0 0 220 118"
        def handle_endtag(self, tag):
            if tag == "section":self.home = False

    page = client.get("/").text;parser = HomeParser();parser.feed(page)
    assert parser.cards == ["column", "section"] and parser.symbols == 2
    assert len(parser.logos) == 2 and all(a["alt"] == "pyBasinSubs" for a in parser.logos)
    assert '<h1>pyBasinSubs</h1>' not in page and '<span class="brand">pyBasinSubs</span>' not in page
    assert '<button data-page="faults" data-home-nav>Fault editor</button>' in page  # header access retained
    assert 'Your files stay on this computer. Nothing is uploaded.' in page
    assert 'Project by Larry Syu-Heng Lai, PhD.' in page and 'AGPL-3.0 or later license' in page
    assert page.index('</main>') < page.index('<footer class="app-credits"')
    import re
    from xml.etree import ElementTree as ET
    icons = re.findall(r'<svg class="card-symbol".*?</svg>', page, re.S)
    schemes = [{"#ab7c85", "#c7a0a7", "#e8d5d9", "#bd919a", "#79515d", "#a37d86"},
               {"#7c98ae", "#a1b8ca", "#d8e3ec", "#66839c", "#48647d", "#8399ac"}]
    for icon, scheme in zip(icons, schemes):
        drawing = ET.fromstring(icon)
        colors = {value for node in drawing.iter() for key, value in node.attrib.items() if key in {"fill", "stroke"}}
        assert colors <= scheme | {"none"}
        fills = {node.get("fill") for node in drawing.iter() if node.get("fill") in scheme}
        assert len(fills) >= 3  # tonal painted layers, not a flat outline
        assert len(drawing.findall(".//path")) >= 2
        assert drawing.findall(".//text")  # original geohistory / restoration labels restored
    assert "older" in icons[0] and "0 Ma" in icons[0]
    assert "unload / restore" in icons[1] and "section distance" in icons[1]
    assert 'apparent throw' in page and 'fixed-trace vertical-shear prototype' in page
    logo = client.get('/static/logo.png')
    assert logo.status_code == 200 and logo.headers['content-type'] == 'image/png'
    assert logo.content == (Path(__file__).resolve().parents[1] / 'image/logo.png').read_bytes()


def test_chrome_styles_keep_credits_in_normal_flow_and_home_responsive(client):
    css = client.get('/static/app.css').text
    footer = css.split('.app-credits {', 1)[1].split('}', 1)[0]
    assert 'position:fixed' not in footer and 'position:absolute' not in footer
    assert 'flex-wrap:wrap' in footer and 'justify-content:space-between' in footer
    assert 'main { flex:1' in css and 'min-height:100dvh' in css
    assert '.home-logo {' in css and '.brand img {' in css
    assert '.card-symbol { display:block; align-self:center; color:#205f70;' in css
    assert '.cards { grid-template-columns:minmax(0,1fr)' in css
    assert '#s_growth_time, #s_thermal_time, #s_prop_time { writing-mode:vertical-lr' in css


def test_wizard_step_markup_and_no_featured_fault_rate_plot(client):
    page = client.get('/').text
    for mode in ['c', 's']:
        ids = [f'{mode}_step{i}' for i in range(1, 6)]
        assert [page.index(f'id="{i}"') for i in ids] == sorted(page.index(f'id="{i}"') for i in ids)
        assert f'id="{mode}_steps" aria-label="Analysis steps"' in page
    assert 'flattening_history' not in page and 'vg1_fault_throw_rates' not in page
    assert 'id="s_open_editor"' in page and 'id="c_open_editor"' not in page
    assert 'id="s_contact"' in page and 'id="s_zero" type="checkbox" checked' in page
    js = client.get('/static/app.js').text
    assert "st.version!==version" in js and "old result discarded" in js
    assert "b.hidden=name!==\"home\"" in js
    assert "invalidate(m,step)" in js


def test_flat_single_dataset_options_and_curve_catalog(client):
    o = client.get("/api/options").json()
    assert o["datasets"][0] == dict(label="Coastal Range N (Lai et al. 2022)", source="Coastal Range N (Lai et al. 2022)", well="")
    assert len(o["datasets"]) == 2 + sum(len(v) for v in o["wells"].values())
    assert not any(d["source"] == "My CSV" for d in o["datasets"])
    assert dict(label="North Viking Graben 30/4-1 (Sodir)", source="North Viking Graben well (Sodir)", well="30/4-1") in o["datasets"]
    assert {d["well"] for d in o["datasets"] if d["source"].startswith("Barents")} == set(o["wells"]["barents_hammerfest"])
    assert o["section_datasets"] == [dict(id="vg1", label="Viking Graben VG1")]
    assert {v["name"] for v in o["porosity_curves"]} == {"mudstone", "sandstone_tw", "marine_avg", "shale", "sandstone", "chalk", "shaly_sandstone"}
    page = client.get("/").text
    assert 'id="c_dataset"' in page and 'id="s_dataset"' in page
    assert 'id="c_src"' not in page and 'id="c_well"' not in page
    for m in ("c", "s"):
        assert f'id="{m}_curve_picker"' in page and f'id="{m}_curve_options"' in page



def test_property_evolution_wizard_markup(client):
    page = client.get("/").text
    assert 'id="c_step6"' in page and "5 · Property evolution" in page
    assert 'id="s_step7"' in page and "6 · Property evolution" in page
    assert 'id="c_prop"' in page and 'id="s_prop"' in page
    assert "Schneider et al. (1996)" in page and "Walderhaug (1996)" in page
    assert 'id="s_prop_var"' in page and 'id="c_prop_time"' in page
    assert 'id="c_prop_note"' not in page and 'id="s_prop_note"' not in page
    assert 'id="c_prop_layers"' not in page and 'id="c_prop_tab"' in page
    assert page.count("Selected timestep layers / depths") == 2
    assert 'id="job_progress"' in page and 'id="job_progress_fill"' in page
    assert "Computing properties…" in page
    js = client.get("/static/app.js").text
    assert "Property evolution" in js
    assert "/api/wizard/column/properties" in js
    assert "/api/wizard/section/properties" in js
    assert "c_prop_tab" in js and "Mechanical porosity" in js and "Pore pressure (MPa)" in js
    assert 'if(r.table)$("c_prop_tab")' not in js
    assert "c_prop_note" not in js and "s_prop_note" not in js
    assert "watchJobProgress" in js and "/api/wizard/progress/" in js
    assert "showJobProgress" in js and "hideJobProgress" in js
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "pybasinsubs/webapp/wizard.py").read_text()
    assert "phi_mechanical" in src and "pore_pressure_mpa" in src
    assert "Property logs ready" in src and "Section properties ready" in src
    assert '@app.get("/api/wizard/progress/{job_id}")' in src
    assert "return guard(wizard_progress)" not in src
