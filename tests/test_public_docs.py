"""Public documentation assets and the overview-only package description."""
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "image").is_dir(), reason="figure assets are source-checkout only")
def test_readme_local_links_and_figures_exist():
    text = (ROOT / "README.md").read_text()
    targets = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
    targets += re.findall(r'src="([^"]+)"', text)
    for target in targets:
        if not target.startswith(("https://", "http://", "#")):
            assert (ROOT / target.split("#", 1)[0]).exists(), target


def test_pypi_description_is_overview_only():
    text = (ROOT / "README-pypi.md").read_text()
    assert not re.search(r"!\[|<img|```|^## (Quick start|Examples|Installation)", text, re.M)
    assert "AGPL-3.0-or-later" in text
    assert "https://doi.org/10.5281/zenodo.23832018" in text
    assert 'readme = "README-pypi.md"' in (ROOT / "pyproject.toml").read_text()
    assert "README-pypi.md" in (ROOT / "MANIFEST.in").read_text()
    assert "conda install" not in text


@pytest.mark.skipif(not (ROOT / "notebooks").is_dir(), reason="notebooks are source-checkout only")
def test_notebook_app_screenshot_links_exist():
    import json

    for number in (5, 9):
        path = next((ROOT / "notebooks").glob(f"pyBasinSubs_{number}_*.ipynb"))
        notebook = json.loads(path.read_text())
        text = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
        for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
            assert (path.parent / target).exists(), target


def test_readme_no_retired_fault_rate_feature():
    text = (ROOT / "README.md").read_text()
    assert "vg1_fault_throw_rates" not in text and "companion export" not in text
    media = ["hero_geohistory_north_sea_30-4-1.webp", "thermal_history_north_sea_30-4-1.webp",
             "vg1_growth_restoration.webp", "vg1_section_thermal_history.webp",
             "pyBasinSubs_column_demo.webp", "pyBasinSubs_section_demo.webp", "app_fault_editor.png"]
    assert [text.index(name) for name in media] == sorted(text.index(name) for name in media)


def test_readme_unreleased_and_animated_1d_figures():
    text = (ROOT / "README.md").read_text()
    assert "This README describes 0.2.0 (unreleased). The published PyPI package is still 0.1.2." not in text
    assert "This README describes 0.3.0 (release draft, unpublished). The published PyPI package is 0.2.0." not in text
    assert "image/app_start.png" not in text
    assert "hero_geohistory_north_sea_30-4-1.png" not in text
    assert "maturity_history_30-4-1.gif" not in text
    assert "1-D Airy backstripping + decompaction" in text
    assert "thermal column + geotherm" in text
    assert "property_logs_north_sea_30-4-1.png" not in text
    assert "vg1_section_porosity_0Ma.png" not in text
    assert "vg1_present_properties.webp" in text
    assert "conda install -c conda-forge pybasinsubs" in text


def test_readme_2d_thermal_and_footwall_wording():
    text = (ROOT / "README.md").read_text()
    assert "explicit growth reconstruction + rates → thermal history" in text
    assert "percent-in-window lines" in text and "histogram" not in text.lower()
    assert "not counted as throw" in text
    assert "not resolved down-dip slip, geological heave or curved-fault balancing" in text
    changelog = (ROOT / "CHANGELOG.md").read_text()
    assert "overlaid unit-colored bars" not in changelog
    assert "footwall_flattening=True" in changelog and "not counted as fault throw" in changelog
