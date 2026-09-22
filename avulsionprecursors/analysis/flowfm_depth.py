"""Sample Delft3D-FM / FlowFM fields for BASED Hm and water-surface slope."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

from avulsionprecursors.sword.base import SwordReach

DEFAULT_DEPTH_VAR = "mesh2d_waterdepth"
DEFAULT_WATERLEVEL_VAR = "mesh2d_s1"
DEFAULT_X_VAR = "mesh2d_face_x"
DEFAULT_Y_VAR = "mesh2d_face_y"
DEFAULT_BACKWATER_LENGTH_M = 1000.0  # legacy fixed window; prefer width-scaled
DEFAULT_SLOPE_WINDOW_WIDTHS = 7.5  # ~5–10 channel widths (geomorphic reach scale)
DEFAULT_SLOPE_MIN = 1e-5
DEFAULT_MIN_SLOPE_WINDOW_M = 200.0
DEFAULT_XS_SAMPLE_STEP_M = 5.0


def _load_face_field(
    nc_path: Path,
    var_name: str,
    time_index: Union[int, str] = -1,
    x_var: str = DEFAULT_X_VAR,
    y_var: str = DEFAULT_Y_VAR,
    require_nonnegative: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load face-center coordinates and a face-centered field.

    ``time_index``:
      - int: that timestep (negative indices allowed)
      - \"max\": maximum over all timesteps at each face
    """
    with Dataset(nc_path) as ds:
        if var_name not in ds.variables:
            raise KeyError(
                f"Variable {var_name!r} not in {nc_path}. "
                f"Available: {list(ds.variables)}"
            )
        for name in (x_var, y_var):
            if name not in ds.variables:
                raise KeyError(f"Coordinate variable {name!r} not in {nc_path}")

        x = np.asarray(ds[x_var][:], dtype=float)
        y = np.asarray(ds[y_var][:], dtype=float)
        var = ds[var_name]
        fill = getattr(var, "_FillValue", None)

        if str(time_index).lower() == "max":
            arr = np.asarray(var[:, :], dtype=float)
            if fill is not None:
                arr = np.where(arr == fill, np.nan, arr)
            if require_nonnegative:
                arr = np.where(arr < 0, np.nan, arr)
            field = np.nanmax(arr, axis=0)
        else:
            t = int(time_index)
            n_time = var.shape[0]
            if t < 0:
                t = n_time + t
            if t < 0 or t >= n_time:
                raise IndexError(
                    f"time_index={time_index} out of range for {n_time} timesteps"
                )
            field = np.asarray(var[t, :], dtype=float)
            if fill is not None:
                field = np.where(field == fill, np.nan, field)
            if require_nonnegative:
                field = np.where(field < 0, np.nan, field)

    if x.shape != field.shape or y.shape != field.shape:
        raise ValueError(
            f"Coordinate/field shape mismatch: x={x.shape}, y={y.shape}, field={field.shape}"
        )
    return x, y, field


def _ls_slope(s_arr: np.ndarray, z_arr: np.ndarray) -> float:
    """Least-squares |dz/ds|; NaN if underdetermined."""
    if s_arr.size < 2:
        return float("nan")
    s_mean = float(np.mean(s_arr))
    z_mean = float(np.mean(z_arr))
    ds = s_arr - s_mean
    denom = float(np.sum(ds * ds))
    if denom < 1e-12:
        return float("nan")
    return abs(float(np.sum(ds * (z_arr - z_mean)) / denom))


def channel_depth_from_flowfm(
    df: pd.DataFrame,
    reach: SwordReach,
    nc_path: Union[str, Path],
    time_index: Union[int, str] = -1,
    sample_step_m: float = 5.0,
    max_neighbor_dist_m: float = 75.0,
    depth_var: str = DEFAULT_DEPTH_VAR,
) -> pd.DataFrame:
    """
    Set ``channel_depth`` (Hm) to the maximum FlowFM water depth sampled along
    each cross-section line.
    """
    nc_path = Path(nc_path)
    if not nc_path.is_file():
        raise FileNotFoundError(f"FlowFM NetCDF not found: {nc_path}")

    x, y, depth = _load_face_field(
        nc_path,
        var_name=depth_var,
        time_index=time_index,
        require_nonnegative=True,
    )
    tree = cKDTree(np.column_stack([x, y]))
    node_geom = {n.node_id: n.cross_section for n in reach.nodes}

    depths: list[float] = []
    n_ok = 0
    for _, row in df.iterrows():
        line = node_geom.get(int(row["node_id"]))
        if line is None or line.is_empty:
            depths.append(np.nan)
            continue

        length = float(line.length)
        if length <= 0:
            depths.append(np.nan)
            continue

        step = max(float(sample_step_m), 1.0)
        npts = max(2, int(np.ceil(length / step)) + 1)
        dists = np.linspace(0.0, length, npts)
        pts = [line.interpolate(float(d)) for d in dists]
        coords = np.array([(p.x, p.y) for p in pts], dtype=float)

        nn_dist, nn_idx = tree.query(coords, k=1)
        wet = (nn_dist <= max_neighbor_dist_m) & np.isfinite(depth[nn_idx])
        vals = depth[nn_idx][wet]
        vals = vals[vals > 0]
        if vals.size == 0:
            depths.append(np.nan)
            continue
        depths.append(float(np.max(vals)))
        n_ok += 1

    out = df.copy()
    out["channel_depth"] = depths
    out["channel_depth_source"] = "flowfm_waterdepth"
    print(
        f"📏 Hm from FlowFM {depth_var} "
        f"(time={time_index!r}): "
        f"{np.nanmin(depths):.2f} – {np.nanmax(depths):.2f} m "
        f"(median {np.nanmedian(depths):.2f} m); "
        f"{n_ok}/{len(depths)} nodes with wet samples"
    )
    return out


def _wse_on_cross_section(
    line: LineString,
    tree: cKDTree,
    wl: np.ndarray,
    sample_step_m: float = DEFAULT_XS_SAMPLE_STEP_M,
    max_neighbor_dist_m: float = 75.0,
    reduce: str = "max",
) -> float:
    """
    Representative water-surface elevation along one cross-section.

    Samples FlowFM face WSE every ``sample_step_m`` along the XS line and
    returns the max or mean of wet (finite, in-range) samples. Using the XS
    (instead of the centerline point) handles centerline–channel misalignment.
    """
    if line is None or line.is_empty:
        return float("nan")
    length = float(line.length)
    if length <= 0:
        return float("nan")

    step = max(float(sample_step_m), 1.0)
    npts = max(2, int(np.ceil(length / step)) + 1)
    dists = np.linspace(0.0, length, npts)
    pts = [line.interpolate(float(d)) for d in dists]
    coords = np.array([(p.x, p.y) for p in pts], dtype=float)
    nn_dist, nn_idx = tree.query(coords, k=1)
    wet = (nn_dist <= max_neighbor_dist_m) & np.isfinite(wl[nn_idx])
    vals = wl[nn_idx][wet]
    if vals.size == 0:
        return float("nan")
    reduce = str(reduce).lower()
    if reduce == "mean":
        return float(np.mean(vals))
    if reduce == "median":
        return float(np.median(vals))
    return float(np.max(vals))


def slope_from_flowfm_waterlevel(
    df: pd.DataFrame,
    reach: SwordReach,
    centerline: LineString,
    nc_path: Union[str, Path],
    time_index: Union[int, str] = -1,
    window_widths: float = DEFAULT_SLOPE_WINDOW_WIDTHS,
    backwater_length_m: Optional[float] = None,
    min_window_m: float = DEFAULT_MIN_SLOPE_WINDOW_M,
    max_neighbor_dist_m: float = 75.0,
    slope_min: float = DEFAULT_SLOPE_MIN,
    waterlevel_var: str = DEFAULT_WATERLEVEL_VAR,
    method: str = "reach_xs",
    xs_wse_reduce: str = "max",
    xs_sample_step_m: float = DEFAULT_XS_SAMPLE_STEP_M,
) -> pd.DataFrame:
    """
    Replace ``slope`` with free-surface slope from FlowFM water level.

    Methods
    -------
    reach_xs (default)
        1) At each cross-section, take max/mean WSE along the XS line.
        2) Reach-average slope = least-squares |dWSE/dstation| using those
           XS WSE values for nodes within a moving window along the centerline
           (``window_widths`` × local width, or fixed ``backwater_length_m``).
    centerline
        Legacy: dense WSE samples on the centerline within the same window.
    """
    method = str(method).lower()
    if method in ("reach_xs", "reach", "xs"):
        return _slope_reach_averaged_from_xs_wse(
            df=df,
            reach=reach,
            centerline=centerline,
            nc_path=nc_path,
            time_index=time_index,
            window_widths=window_widths,
            backwater_length_m=backwater_length_m,
            min_window_m=min_window_m,
            max_neighbor_dist_m=max_neighbor_dist_m,
            slope_min=slope_min,
            waterlevel_var=waterlevel_var,
            xs_wse_reduce=xs_wse_reduce,
            xs_sample_step_m=xs_sample_step_m,
        )
    if method in ("centerline", "cl", "legacy"):
        return _slope_from_centerline_wse(
            df=df,
            reach=reach,
            centerline=centerline,
            nc_path=nc_path,
            time_index=time_index,
            window_widths=window_widths,
            backwater_length_m=backwater_length_m,
            min_window_m=min_window_m,
            max_neighbor_dist_m=max_neighbor_dist_m,
            slope_min=slope_min,
            waterlevel_var=waterlevel_var,
        )
    raise ValueError(
        f"Unknown FlowFM slope method {method!r}; use 'reach_xs' or 'centerline'"
    )


def _slope_window_m(
    width: float,
    window_widths: float,
    backwater_length_m: Optional[float],
    min_window_m: float,
) -> float:
    fixed_lb = float(backwater_length_m) if backwater_length_m is not None else None
    if fixed_lb is not None and fixed_lb > 0:
        return fixed_lb
    if np.isfinite(width) and width > 0 and window_widths > 0:
        return max(float(min_window_m), float(window_widths) * width)
    return float(min_window_m)


def _slope_reach_averaged_from_xs_wse(
    df: pd.DataFrame,
    reach: SwordReach,
    centerline: LineString,
    nc_path: Union[str, Path],
    time_index: Union[int, str],
    window_widths: float,
    backwater_length_m: Optional[float],
    min_window_m: float,
    max_neighbor_dist_m: float,
    slope_min: float,
    waterlevel_var: str,
    xs_wse_reduce: str,
    xs_sample_step_m: float,
) -> pd.DataFrame:
    """
    Reach-averaged Sm from XS-representative WSE vs. centerline station.
    """
    nc_path = Path(nc_path)
    if not nc_path.is_file():
        raise FileNotFoundError(f"FlowFM NetCDF not found: {nc_path}")
    if centerline is None or centerline.is_empty:
        raise ValueError("Valid centerline LineString is required for FlowFM slope")

    x, y, wl = _load_face_field(
        nc_path,
        var_name=waterlevel_var,
        time_index=time_index,
        require_nonnegative=False,
    )
    tree = cKDTree(np.column_stack([x, y]))
    node_by_id = {n.node_id: n for n in reach.nodes}
    line_length = float(centerline.length)

    out = df.copy()
    out["slope_dem"] = out["slope"].astype(float)

    # Pass 1: station + XS WSE for every node
    stations: list[float] = []
    wse_xs: list[float] = []
    widths: list[float] = []
    for _, row in out.iterrows():
        nid = int(row["node_id"])
        node = node_by_id.get(nid)
        width = float(row["width"]) if pd.notna(row.get("width")) else float("nan")
        widths.append(width)

        pt = None
        xs_line = None
        if node is not None:
            xs_line = node.cross_section
            if node.geometry is not None and not node.geometry.is_empty:
                pt = node.geometry
            elif xs_line is not None and not xs_line.is_empty:
                pt = xs_line.interpolate(0.5, normalized=True)

        if pt is None or line_length <= 0:
            stations.append(float("nan"))
            wse_xs.append(float("nan"))
            continue

        s = float(centerline.project(Point(pt.x, pt.y)))
        stations.append(s)
        wse_xs.append(
            _wse_on_cross_section(
                xs_line,
                tree,
                wl,
                sample_step_m=xs_sample_step_m,
                max_neighbor_dist_m=max_neighbor_dist_m,
                reduce=xs_wse_reduce,
            )
        )

    out["wse_xs"] = wse_xs
    out["station_m"] = stations
    s_all = np.asarray(stations, dtype=float)
    z_all = np.asarray(wse_xs, dtype=float)

    # Pass 2: moving-window LS slope from neighboring XS WSE values
    slopes: list[float] = []
    sources: list[str] = []
    windows: list[float] = []
    n_ok = 0
    n_fallback = 0

    for i, (_, row) in enumerate(out.iterrows()):
        dem_s = float(row["slope_dem"]) if pd.notna(row["slope_dem"]) else float("nan")
        win = _slope_window_m(
            widths[i], window_widths, backwater_length_m, min_window_m
        )
        windows.append(win)
        si = s_all[i]

        if not np.isfinite(si) or win <= 0:
            slopes.append(dem_s)
            sources.append("dem_fallback")
            n_fallback += 1
            continue

        half = 0.5 * win
        in_win = (
            np.isfinite(s_all)
            & np.isfinite(z_all)
            & (s_all >= si - half)
            & (s_all <= si + half)
        )
        if int(np.count_nonzero(in_win)) < 3:
            slopes.append(dem_s)
            sources.append("dem_fallback")
            n_fallback += 1
            continue

        slope = _ls_slope(s_all[in_win], z_all[in_win])
        if not np.isfinite(slope):
            slopes.append(dem_s)
            sources.append("dem_fallback")
            n_fallback += 1
            continue

        if slope < slope_min:
            slope = slope_min
        slopes.append(float(slope))
        sources.append("flowfm_reach_xs")
        n_ok += 1

    out["slope"] = slopes
    out["slope_source"] = sources
    out["slope_window_m"] = windows
    s_arr = np.asarray(slopes, dtype=float)
    w_arr = np.asarray(windows, dtype=float)
    if backwater_length_m is not None:
        win_desc = f"fixed Lb={float(backwater_length_m):.0f} m"
    else:
        win_desc = (
            f"{window_widths:g}×width "
            f"(windows {np.nanmin(w_arr):.0f}–{np.nanmax(w_arr):.0f} m, "
            f"median {np.nanmedian(w_arr):.0f} m)"
        )
    print(
        f"📐 Reach-avg Sm from FlowFM {waterlevel_var} "
        f"(XS {xs_wse_reduce} WSE → WSE~station, {win_desc}, time={time_index!r}): "
        f"{np.nanmin(s_arr):.2e} – {np.nanmax(s_arr):.2e} "
        f"(median {np.nanmedian(s_arr):.2e}); "
        f"{n_ok} FlowFM / {n_fallback} DEM fallback; "
        f"wet XS {np.isfinite(z_all).sum()}/{len(z_all)}"
    )
    return out


def _slope_from_centerline_wse(
    df: pd.DataFrame,
    reach: SwordReach,
    centerline: LineString,
    nc_path: Union[str, Path],
    time_index: Union[int, str] = -1,
    window_widths: float = DEFAULT_SLOPE_WINDOW_WIDTHS,
    backwater_length_m: Optional[float] = None,
    min_window_m: float = DEFAULT_MIN_SLOPE_WINDOW_M,
    max_neighbor_dist_m: float = 75.0,
    slope_min: float = DEFAULT_SLOPE_MIN,
    waterlevel_var: str = DEFAULT_WATERLEVEL_VAR,
) -> pd.DataFrame:
    """
    Legacy free-surface slope: dense centerline WSE samples in a moving window.
    """
    nc_path = Path(nc_path)
    if not nc_path.is_file():
        raise FileNotFoundError(f"FlowFM NetCDF not found: {nc_path}")
    if centerline is None or centerline.is_empty:
        raise ValueError("Valid centerline LineString is required for FlowFM slope")

    x, y, wl = _load_face_field(
        nc_path,
        var_name=waterlevel_var,
        time_index=time_index,
        require_nonnegative=False,
    )
    tree = cKDTree(np.column_stack([x, y]))
    node_by_id = {n.node_id: n for n in reach.nodes}

    line_length = float(centerline.length)

    out = df.copy()
    out["slope_dem"] = out["slope"].astype(float)

    slopes: list[float] = []
    sources: list[str] = []
    windows: list[float] = []
    n_ok = 0
    n_fallback = 0

    for _, row in out.iterrows():
        nid = int(row["node_id"])
        node = node_by_id.get(nid)
        dem_s = float(row["slope_dem"]) if pd.notna(row["slope_dem"]) else float("nan")

        pt = None
        if node is not None and node.geometry is not None and not node.geometry.is_empty:
            pt = node.geometry
        elif node is not None and node.cross_section is not None:
            pt = node.cross_section.interpolate(0.5, normalized=True)

        width = float(row["width"]) if pd.notna(row.get("width")) else float("nan")
        win = _slope_window_m(width, window_widths, backwater_length_m, min_window_m)

        def _fallback() -> None:
            nonlocal n_fallback
            slopes.append(dem_s)
            sources.append("dem_fallback")
            windows.append(win)
            n_fallback += 1

        if pt is None or win <= 0 or line_length <= 0:
            _fallback()
            continue

        half = 0.5 * win
        s = float(centerline.project(Point(pt.x, pt.y)))
        s_start = max(0.0, s - half)
        s_end = min(line_length, s + half)
        span = s_end - s_start
        if span < 1e-3:
            _fallback()
            continue

        step = max(5.0, min(25.0, span / 40.0))
        distances = np.arange(s_start, s_end + 1e-9, step)
        if len(distances) < 2:
            distances = np.linspace(s_start, s_end, max(8, int(span / 5.0) + 1))

        pts = [centerline.interpolate(float(si)) for si in distances]
        coords = np.array([(p.x, p.y) for p in pts], dtype=float)
        nn_dist, nn_idx = tree.query(coords, k=1)
        ok = (nn_dist <= max_neighbor_dist_m) & np.isfinite(wl[nn_idx])
        if int(np.count_nonzero(ok)) < 3:
            _fallback()
            continue

        slope = _ls_slope(
            np.asarray(distances, dtype=float)[ok],
            wl[nn_idx][ok],
        )
        if not np.isfinite(slope):
            _fallback()
            continue

        if slope < slope_min:
            slope = slope_min
        slopes.append(float(slope))
        sources.append("flowfm_centerline")
        windows.append(win)
        n_ok += 1

    out["slope"] = slopes
    out["slope_source"] = sources
    out["slope_window_m"] = windows
    s_arr = np.asarray(slopes, dtype=float)
    w_arr = np.asarray(windows, dtype=float)
    if backwater_length_m is not None:
        win_desc = f"fixed Lb={float(backwater_length_m):.0f} m"
    else:
        win_desc = (
            f"{window_widths:g}×width "
            f"(windows {np.nanmin(w_arr):.0f}–{np.nanmax(w_arr):.0f} m, "
            f"median {np.nanmedian(w_arr):.0f} m)"
        )
    print(
        f"📐 Slope from FlowFM {waterlevel_var} centerline WSE~station "
        f"({win_desc}, time={time_index!r}): "
        f"{np.nanmin(s_arr):.2e} – {np.nanmax(s_arr):.2e} "
        f"(median {np.nanmedian(s_arr):.2e}); "
        f"{n_ok} FlowFM / {n_fallback} DEM fallback"
    )
    return out
