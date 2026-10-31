# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Result container of the 2-D fault restorations."""
from __future__ import annotations
from dataclasses import dataclass, field, fields
import numpy as np


@dataclass
class Restoration2D:
    """Restored section at every age step (returned by ``restore``, ``restore_faulted``, ``restore_kinematic``).

    Arrays: top/base (nt, nu, nx) depth of unit tops/bases (NaN = stripped), seafloor (nt, nx).
    ``extras`` holds method-specific outputs (cantilever: u_fault, slip, rest; kinematic: steps,
    overlap_m, gap_m, listric_dips, listric_z0, alpha). Dict-style access is kept for backward
    compatibility: ``r["top"]``, ``r["steps"]``, ``"slip" in r``, ``r.get(...)``, ``r.keys()``.
    """
    ages: np.ndarray
    x: np.ndarray
    units: list
    top: np.ndarray
    base: np.ndarray
    seafloor: np.ndarray
    method: str = "cantilever"
    extras: dict = field(default_factory=dict)

    def _names(self):
        return [f.name for f in fields(self) if f.name != "extras"]

    def __getitem__(self, k):
        if k in self._names():
            return getattr(self, k)
        return self.extras[k]

    def __setitem__(self, k, v):
        if k in self._names():
            setattr(self, k, v)
        else:
            self.extras[k] = v

    def __contains__(self, k):
        return k in self._names() or k in self.extras

    def __getattr__(self, k):                     # extras as attributes (r.steps, r.slip)
        if k != "extras" and "extras" in self.__dict__ and k in self.__dict__["extras"]:
            return self.__dict__["extras"][k]
        raise AttributeError(k)

    def get(self, k, default=None):
        return self[k] if k in self else default

    def keys(self):
        return self._names() + list(self.extras)

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def surface(self, i, unit=None):
        """Top of `unit` (default: the youngest remaining unit, i.e. the restored datum) at step i."""
        return self.top[i][i if unit is None else unit]
