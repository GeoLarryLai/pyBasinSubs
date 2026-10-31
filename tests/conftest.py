# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Test configuration.

The Miller et al. (2020) sea-level curve is CC BY-NC 4.0 and is not shipped; it is
downloaded and cached on first use (pybasinsubs.sealevel.miller2020_path). When the
download is not possible (offline CI), a synthetic curve with the same format
(tests/fixtures/synthetic_sealevel.csv; generated, contains no Miller data) is used and
the tests that compare against R outputs computed with the real curve are skipped.
"""
import os
from pathlib import Path
import pytest

SYNTH = Path(__file__).parent / "fixtures" / "synthetic_sealevel.csv"


def _miller_available():
    if os.environ.get("PYBASINSUBS_MILLER2020"):
        return True
    from pybasinsubs.sealevel import miller2020_path
    try:
        miller2020_path()
        return True
    except Exception:
        return False


MILLER_REAL = _miller_available()
if not MILLER_REAL:
    os.environ["PYBASINSUBS_MILLER2020"] = str(SYNTH)


def pytest_configure(config):
    config.addinivalue_line("markers", "real_miller: needs the real Miller et al. (2020) curve")
    config.addinivalue_line("markers", "slow: long-running (notebook execution); run with  pytest --runslow")


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run tests marked slow")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="slow; use --runslow")
        for it in items:
            if "slow" in it.keywords:
                it.add_marker(skip_slow)
    if MILLER_REAL:
        return
    skip = pytest.mark.skip(reason="Miller et al. (2020) curve not downloadable; synthetic fixture in use")
    for it in items:
        if "real_miller" in it.keywords:
            it.add_marker(skip)
