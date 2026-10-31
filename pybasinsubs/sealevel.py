# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Eustatic sea level (Miller et al., 2020, Science Advances 6:eaaz1346).

The curve covers 0-66.6 Ma. It is NOT shipped with the package: the Miller et al.
(2020) data are published under CC BY-NC 4.0, which is incompatible with the
AGPL-3.0-or-later licence of this package. On first use it is downloaded from the
BasinSubs repository (github.com/GeoLarryLai/BasinSubs, file
Sealevel/SeaLevel_Miller2020.csv), checked against a SHA-256 and cached in
``~/.cache/pybasinsubs`` (env PYBASINSUBS_CACHE). Offline: point env
PYBASINSUBS_MILLER2020 at a local copy. Non-commercial use only; cite
Miller, K.G. et al. (2020) Sci. Adv. 6, eaaz1346. Older ages return 0 +- ``pre_cenozoic_err``
(an explicit assumption; replace with e.g. Haq 2014/2018 if you have it).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import os
import shutil
import urllib.request
import hashlib
from ._paths import cache_dir

MILLER_URL = "https://raw.githubusercontent.com/GeoLarryLai/BasinSubs/main/Sealevel/SeaLevel_Miller2020.csv"
MILLER_SHA256 = "8bbe0cb2881245834e05a08f96bb657386faade7f7f42bdc73c93a5c90c95961"
MILLER_FILE = "SeaLevel_Miller2020.csv"

_CURVE = None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def miller2020_path(download=True):
    """Local path of the Miller et al. (2020) CSV, downloading and caching it on first use."""
    env = os.environ.get("PYBASINSUBS_MILLER2020")
    if env:
        return env
    p = os.path.join(cache_dir(), MILLER_FILE)
    if os.path.exists(p) and _sha256(p) == MILLER_SHA256:
        return p
    if not download:
        raise FileNotFoundError(p)
    tmp = p + ".part"
    try:
        with urllib.request.urlopen(MILLER_URL, timeout=30) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
    except Exception as e:                                   # pragma: no cover - network
        raise RuntimeError(f"Could not download the Miller et al. (2020) sea-level curve ({e}). "
                           f"Download {MILLER_URL} manually and set PYBASINSUBS_MILLER2020=/path/to/file.") from e
    if _sha256(tmp) != MILLER_SHA256:
        os.remove(tmp)
        raise RuntimeError("Checksum mismatch for the downloaded Miller et al. (2020) curve.")
    os.replace(tmp, p)
    print("pybasinsubs: downloaded Miller et al. (2020) sea level (CC BY-NC 4.0, non-commercial use; "
          "cite Sci. Adv. 6, eaaz1346) to " + p)
    return p


def miller2020():


    """Miller et al. (2020) global sea-level curve (0-66.6 Ma), as a DataFrame (sea_level_m, age_Ma), cached."""
    global _CURVE
    if _CURVE is None:
        c = pd.read_csv(miller2020_path(), encoding="utf-8-sig")
        c.columns = ["sea_level_m", "age_Ma"]
        _CURVE = c.sort_values("age_Ma").reset_index(drop=True)
    return _CURVE


def sea_level_at(age, age_err=None, window_min=0.01, pre_cenozoic_err=100.0):
    """Mean and standard error of the Miller curve within age +- (err + window).

    Same rule as BasinSubs ``SL_Unit`` (mean, sd/sqrt(n)). Returns arrays.
    """
    c = miller2020()
    a = np.atleast_1d(np.asarray(age, float))
    e = np.zeros_like(a) if age_err is None else np.atleast_1d(np.asarray(age_err, float))
    m = np.zeros_like(a); s = np.zeros_like(a)
    amax = c.age_Ma.max()
    for i, (ai, ei) in enumerate(zip(a, e)):
        if ai > amax:
            m[i], s[i] = 0.0, pre_cenozoic_err
            continue
        sel = c.sea_level_m[(c.age_Ma >= ai - ei - window_min) & (c.age_Ma <= ai + ei + window_min)]
        if len(sel) < 2:
            m[i] = np.interp(ai, c.age_Ma, c.sea_level_m); s[i] = 10.0
        else:
            m[i] = sel.mean(); s[i] = sel.std(ddof=1) / np.sqrt(len(sel))
    return m, s
