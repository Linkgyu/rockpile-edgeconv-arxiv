"""PSD utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cumulative_psd(diameter_m: np.ndarray, proxy_volume_m3: np.ndarray) -> pd.DataFrame:
    diameter = np.asarray(diameter_m, dtype=float)
    volume = np.asarray(proxy_volume_m3, dtype=float)
    order = np.argsort(diameter)
    diameter = diameter[order]
    volume = volume[order]
    frac = volume / np.clip(volume.sum(), 1e-12, None)
    return pd.DataFrame({
        "diameter_m": diameter,
        "diameter_mm": diameter * 1000,
        "proxy_volume_m3": volume,
        "volume_fraction": frac,
        "cumulative_passing_pct": np.cumsum(frac) * 100,
    })


def percentile_size(psd_df: pd.DataFrame, passing_pct: float) -> float:
    return float(np.interp(passing_pct, psd_df["cumulative_passing_pct"], psd_df["diameter_mm"]))
