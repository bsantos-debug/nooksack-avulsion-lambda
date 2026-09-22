from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rasterio

from avulsionprecursors.analysis.based import _extract_power_law_ab
from avulsionprecursors.sword.base import SwordReach


class BASEDAnalyzer:
    """
    BASED-style metrics using channel depth from a bathymetry raster along each
    cross-section (instead of XGBoost depth).
    """

    min_slope: float = 1e-6

    def __init__(
        self,
        bathymetry_path: str,
        params_path: Optional[str] = None,
        bathymetry_is_depth: bool = True,
        bathymetry_units: str = "m",
    ) -> None:
        self.bathymetry_path = Path(bathymetry_path)
        if not self.bathymetry_path.is_file():
            raise FileNotFoundError(f"Bathymetry raster not found: {self.bathymetry_path}")
        self.bathymetry_is_depth = bathymetry_is_depth
        self.bathymetry_units = bathymetry_units.lower()
        self.params = None
        self._a = self._b = None
        if params_path and os.path.isfile(params_path):
            with open(params_path, "rb") as f:
                raw = pickle.load(f)
            self.params = raw
            self._a, self._b = _extract_power_law_ab(raw)

    def _nodes_to_dataframe(self, reach: SwordReach) -> pd.DataFrame:
        rows = []
        for n in reach.nodes:
            if n.cross_section is None:
                continue
            if n.slope is None:
                continue
            width = float(n.width) if n.width is not None else np.nan
            rows.append(
                {
                    "node_id": n.node_id,
                    "reach_id": n.reach_id,
                    "dist_out": n.dist_out,
                    "width": width,
                    "slope": float(n.slope),
                    "elevation": float(n.elevation) if n.elevation is not None else np.nan,
                }
            )
        return pd.DataFrame(rows)

    def _calculate_discharge(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._a is None or self._b is None:
            return df
        if "discharge_value" not in df.columns:
            raise ValueError("Discharge value missing from data (need 'discharge_value' column).")
        q = df["discharge_value"].astype(float)
        out = df.copy()
        out["corrected_discharge"] = (q / self._a) ** (1.0 / self._b)
        return out

    def _calculate_depth_from_bathymetry(
        self, df: pd.DataFrame, reach: SwordReach
    ) -> pd.DataFrame:
        """
        Set channel depth from the bathymetry/topobathy raster along each cross-section.

        - If ``bathymetry_is_depth``: raster values are already depths; use the max
          within the channel window.
        - Else: raster values are elevations; depth = max(z) - min(z) within the
          channel window (bankfull-style depth from the topobathy profile).

        The channel window is centered on the labeled channel position (or the
        cross-section midpoint) with half-width = 0.5 * channel width.
        """
        node_geom = {n.node_id: n.cross_section for n in reach.nodes}
        ft_to_m = 0.3048
        depths = []
        with rasterio.open(self.bathymetry_path) as dem:
            for _, row in df.iterrows():
                line = node_geom.get(int(row["node_id"]))
                if line is None or line.is_empty:
                    depths.append(np.nan)
                    continue

                length = float(line.length)
                if length <= 0:
                    depths.append(np.nan)
                    continue

                npts = max(2, int(length) + 1)
                dists = np.linspace(0.0, length, npts)

                if "channel_dist_along" in row and pd.notna(row["channel_dist_along"]):
                    center = float(row["channel_dist_along"])
                else:
                    center = 0.5 * length
                half_w = 0.5 * float(row["width"]) if pd.notna(row.get("width")) else 25.0
                half_w = max(half_w, 5.0)
                in_channel = (dists >= center - half_w) & (dists <= center + half_w)
                if not np.any(in_channel):
                    in_channel = np.ones_like(dists, dtype=bool)

                pts = [line.interpolate(float(d)) for d in dists[in_channel]]
                coords = [(p.x, p.y) for p in pts]
                samples = []
                for v in dem.sample(coords):
                    z = float(v[0])
                    if dem.nodata is not None and z == dem.nodata:
                        continue
                    if np.isnan(z):
                        continue
                    samples.append(z)

                if not samples:
                    depths.append(np.nan)
                    continue

                arr = np.asarray(samples, dtype=float)
                if self.bathymetry_is_depth:
                    d = float(np.nanmax(arr))
                else:
                    d = float(np.nanmax(arr) - np.nanmin(arr))
                    if d <= 0:
                        depths.append(np.nan)
                        continue

                if self.bathymetry_units == "ft":
                    d *= ft_to_m
                depths.append(d)

        out = df.copy()
        out["channel_depth"] = depths
        return out

    def _calculate_ridge_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Har = ridge crest − floodplain elevation.
        SAR = Har / intervening horizontal distance (ridge to floodplain).
        """
        required = (
            "ridge1_dist_along",
            "ridge1_elevation",
            "floodplain1_dist_along",
            "floodplain1_elevation",
        )
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Missing label column {c!r}.")

        out = df.copy()
        eps = 1e-6

        out["Har_ridge1"] = out["ridge1_elevation"] - out["floodplain1_elevation"]
        run1 = (
            out["ridge1_dist_along"] - out["floodplain1_dist_along"]
        ).abs().clip(lower=eps)
        out["SAR_ridge1"] = out["Har_ridge1"].abs() / run1

        if (
            "ridge2_dist_along" in out.columns
            and "ridge2_elevation" in out.columns
            and "floodplain2_dist_along" in out.columns
            and "floodplain2_elevation" in out.columns
        ):
            out["Har_ridge2"] = out["ridge2_elevation"] - out["floodplain2_elevation"]
            run2 = (
                out["ridge2_dist_along"] - out["floodplain2_dist_along"]
            ).abs().clip(lower=eps)
            out["SAR_ridge2"] = out["Har_ridge2"].abs() / run2
        else:
            out["Har_ridge2"] = np.nan
            out["SAR_ridge2"] = np.nan

        out["Har_mean"] = np.nanmean(
            np.vstack(
                [
                    out["Har_ridge1"].to_numpy(dtype=float),
                    out["Har_ridge2"].to_numpy(dtype=float),
                ]
            ),
            axis=0,
        )
        return out

    def _calculate_gamma(self, df: pd.DataFrame) -> pd.DataFrame:
        if "SAR_ridge1" not in df.columns:
            raise ValueError("Run _calculate_ridge_parameters first.")
        out = df.copy()
        sm = out["slope"].clip(lower=self.min_slope).astype(float)
        g1 = out["SAR_ridge1"].astype(float) / sm
        g2 = out["SAR_ridge2"].astype(float) / sm
        stack = np.vstack([g1.to_numpy(), g2.to_numpy()])
        out["gamma_mean"] = np.nanmean(stack, axis=0)
        return out

    def _calculate_superelevation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Superelevation β = Har / Hm.

        Har = ridge crest − floodplain; Hm = ridge crest − channel bed.
        Computed per bank, then averaged.
        """
        if "Har_ridge1" not in df.columns:
            raise ValueError("Run _calculate_ridge_parameters first (need Har).")
        for col in ("ridge1_elevation", "channel_elevation"):
            if col not in df.columns:
                raise ValueError(f"Missing label column {col!r} for Hm.")

        out = df.copy()

        # Prefer explicit per-bank Hm if already set; else ridge − channel
        if "Hm_ridge1" in out.columns:
            hm1 = out["Hm_ridge1"].astype(float)
        else:
            hm1 = (out["ridge1_elevation"] - out["channel_elevation"]).astype(float)
            out["Hm_ridge1"] = hm1

        if "Hm_ridge2" in out.columns:
            hm2 = out["Hm_ridge2"].astype(float)
        elif "ridge2_elevation" in out.columns:
            hm2 = (out["ridge2_elevation"] - out["channel_elevation"]).astype(float)
            out["Hm_ridge2"] = hm2
        else:
            hm2 = pd.Series(np.nan, index=out.index)
            out["Hm_ridge2"] = hm2

        hm1 = hm1.clip(lower=self.min_slope)
        hm2 = hm2.clip(lower=self.min_slope)
        out["channel_depth"] = np.nanmean(
            np.vstack([hm1.to_numpy(dtype=float), hm2.to_numpy(dtype=float)]),
            axis=0,
        )

        b1 = out["Har_ridge1"] / hm1
        b2 = out["Har_ridge2"] / hm2 if "Har_ridge2" in out.columns else pd.Series(np.nan, index=out.index)
        stack = np.vstack([b1.to_numpy(dtype=float), b2.to_numpy(dtype=float)])
        out["superelevation_mean"] = np.nanmean(stack, axis=0)
        if "Har_mean" not in out.columns:
            out["Har_mean"] = np.nanmean(
                np.vstack(
                    [
                        out["Har_ridge1"].to_numpy(dtype=float),
                        out["Har_ridge2"].to_numpy(dtype=float),
                    ]
                ),
                axis=0,
            )
        return out

    def _add_quality_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["flag_valid_lambda"] = out["lambda"].notna() & (out["lambda"] > 0)
        return out
