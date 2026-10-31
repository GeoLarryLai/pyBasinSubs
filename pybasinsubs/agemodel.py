# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Piecewise-linear age-height models (BasinSubs ``Age PB dSL`` approach).

Each segment is an OLS fit age = a + b h over a range of dated horizons;
predictions carry the standard error of the fitted mean (R ``predict(se.fit=TRUE)``).
"""
from __future__ import annotations
import numpy as np
from .legacy import _LM


class PiecewiseAgeModel:
    """Piecewise-linear age-height model: one OLS line per segment, breaks at line intersections."""
    def __init__(self, heights, ages, segments):
        """segments: list of (i0, i1) slices into heights/ages (i1 exclusive)."""
        h = np.asarray(heights, float); a = np.asarray(ages, float)
        self.models = [_LM(h[i0:i1], a[i0:i1]) for i0, i1 in segments]
        self.breaks = [abs((m2.intercept - m1.intercept) / (m2.slope - m1.slope))
                       for m1, m2 in zip(self.models[:-1], self.models[1:])]

    def segment(self, h):
        """Index of the segment for height h (segments ordered top -> bottom)."""
        for k, b in enumerate(self.breaks):
            if h >= b:
                return k
        return len(self.models) - 1

    def predict(self, h):
        h = np.atleast_1d(np.asarray(h, float))
        out = np.full((len(h), 2), np.nan)
        for i, hi in enumerate(h):
            if np.isnan(hi):
                continue
            f, se = self.models[self.segment(hi)].predict([hi])
            out[i] = f[0], se[0]
        return out[:, 0], out[:, 1]
