"""Labelled synthetic abdomen so the UI works without patient data or a server."""
from __future__ import annotations

import numpy as np


def make(nx: int = 192, ny: int = 160, nz: int = 80):
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    cx, cy = nx / 2, ny / 2
    rng = np.random.default_rng(7)
    ct = np.full((nx, ny, nz), -1000, np.int16)
    mask = np.zeros((nx, ny, nz), np.uint8)
    for k in range(nz):
        s = 1 + 0.08 * np.sin(k / nz * np.pi)              # body slightly wider mid-segment
        r = np.sqrt(((x - cx) / (0.42 * nx * s)) ** 2 + ((y - cy) / (0.40 * ny * s)) ** 2)
        sl_ct = np.full((nx, ny), -1000, np.int16)
        sl_m = np.zeros((nx, ny), np.uint8)
        sl_ct[r < 1.0] = -100; sl_m[r < 1.0] = 2                     # subcutaneous fat
        sl_ct[r < 0.82] = 40;  sl_m[r < 0.82] = 1                    # muscle wall
        sl_ct[r < 0.70] = -95; sl_m[r < 0.70] = 3                    # visceral fat
        organs = (((x - cx * 0.85) / (0.17 * nx)) ** 2 + ((y - cy) / (0.15 * ny)) ** 2 < 1) | \
                 (((x - cx * 1.25) / (0.10 * nx)) ** 2 + ((y - cy * 1.05) / (0.10 * ny)) ** 2 < 1)
        sl_ct[organs & (r < 0.7)] = 45; sl_m[organs & (r < 0.7)] = 0
        psoas = ((((x - cx * 0.80) / (0.07 * nx)) ** 2 + ((y - cy * 1.45) / (0.08 * ny)) ** 2 < 1) |
                 (((x - cx * 1.20) / (0.07 * nx)) ** 2 + ((y - cy * 1.45) / (0.08 * ny)) ** 2 < 1))
        sl_ct[psoas] = 42; sl_m[psoas] = 1
        spine = ((x - cx) / (0.06 * nx)) ** 2 + ((y - cy * 1.55) / (0.07 * ny)) ** 2 < 1
        sl_ct[spine] = 700; sl_m[spine] = 0
        imat = (sl_m == 1) & (rng.random((nx, ny)) < 0.04)
        sl_ct[imat] = -60; sl_m[imat] = 4
        ct[:, :, k] = sl_ct + rng.integers(-12, 13, (nx, ny), dtype=np.int16) * (sl_ct > -900)
        mask[:, :, k] = sl_m
    affine = np.diag([0.78, 0.78, 2.5, 1.0])
    affine[:3, 3] = [-nx * 0.39, -ny * 0.39, 0]
    return ct, mask, affine, (0.78, 0.78, 2.5)
