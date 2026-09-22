from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from avulsionprecursors.sword.base import SwordReach


def _extract_power_law_ab(params: Any) -> tuple[float, float]:
    if isinstance(params, dict):
        if "a" in params and "b" in params:
            return float(params["a"]), float(params["b"])
        for k in ("inverse_power_law", "power_law", "coefficients"):
            inner = params.get(k)
            if isinstance(inner, dict) and "a" in inner and "b" in inner:
                return float(inner["a"]), float(inner["b"])
    # Some pickles store [a, b] as an array or sequence
    if isinstance(params, (list, tuple, np.ndarray)):
        arr = np.asarray(params).ravel()
        if arr.size >= 2:
            return float(arr[0]), float(arr[1])
    raise ValueError(
        "Inverse power-law pickle must expose coefficients 'a' and 'b' "
        "(top-level or under a nested dict, or as a length-2 array)."
    )


class BASEDAnalyzer:
    """
    BASED-style hydraulic geometry and avulsion metrics (lambda).

    Expects an XGBoost depth model and inverse power-law parameters for discharge
    correction, matching the workflow described in BASED_REQUIREMENTS.md.
    """

    min_slope: float = 1e-6

    def __init__(
        self,
        model_path: str,
        params_path: str,
    ) -> None:
        self.model_path = Path(model_path)
        self.params_path = Path(params_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"XGBoost model not found: {self.model_path}")
        if not self.params_path.is_file():
            raise FileNotFoundError(f"Parameter pickle not found: {self.params_path}")
        with open(self.params_path, "rb") as f:
            raw_params = pickle.load(f)
        self._a, self._b = _extract_power_law_ab(raw_params)
        self.params = raw_params
        self._model = joblib.load(self.model_path)

    def _nodes_to_dataframe(self, reach: SwordReach) -> pd.DataFrame:
        rows = []
        for n in reach.nodes:
            if n.cross_section is None:
                continue
            if n.width is None or n.slope is None:
                continue
            rows.append(
                {
                    "node_id": n.node_id,
                    "reach_id": n.reach_id,
                    "dist_out": n.dist_out,
                    "width": float(n.width),
                    "slope": float(n.slope),
                    "elevation": float(n.elevation) if n.elevation is not None else np.nan,
                }
            )
        return pd.DataFrame(rows)

    def _calculate_discharge(self, df: pd.DataFrame) -> pd.DataFrame:
        if "discharge_value" not in df.columns:
            raise ValueError("Discharge value missing from data (need 'discharge_value' column).")
        q = df["discharge_value"].astype(float)
        df = df.copy()
        df["corrected_discharge"] = (q / self._a) ** (1.0 / self._b)
        return df

    def _predict_depth(self, df: pd.DataFrame) -> pd.DataFrame:
        need = ("width", "slope", "corrected_discharge")
        for c in need:
            if c not in df.columns:
                raise ValueError(f"Missing column {c!r} required for depth prediction.")
        X = np.column_stack(
            [
                df["width"].astype(float).to_numpy(),
                df["slope"].astype(float).to_numpy(),
                df["corrected_discharge"].astype(float).to_numpy(),
            ]
        )
        out = df.copy()
        out["XGB_depth"] = self._model.predict(X)
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
