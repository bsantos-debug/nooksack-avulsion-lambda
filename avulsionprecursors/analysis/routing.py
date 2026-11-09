"""Softmax avulsion routing and the slope of the most frequent paths.

Adapted from ``figure_plotting/walk_funcs.py`` in Jake Gearon's
`rulesofriveravulsion` repository
(https://github.com/jameshgrn/rulesofriveravulsion), companion code to
Gearon et al. (2024), *Nature*, https://doi.org/10.1038/s41586-024-07964-2.

The walk matches that implementation. From the current cell the eight
neighbors get a transition probability

    momentum = alpha * (1 + cosine_similarity) / 2
    weight   = (1 - beta) * momentum + beta * (theta * pixel_slope)
    P        = softmax(weight)

``pixel_slope`` is rise / pixel distance (1 or √2), positive downhill, as
in the original walk. The slope returned for Λ is the same rise divided by
map distance, averaged over steps that land in cells whose visit count
is at least ``frequency_threshold`` times the busiest cell (Figure 4 uses
0.075 on the normalized occupation). Defaults ``theta=10``, ``alpha=1``,
``beta=0.5`` match the Figure 4
tool and the paper's ω = 0.5, ϕ = 10, ψ = 1.

If ``outward`` is set (the pixel vector from the channel to the seed), steps
back toward the channel are refused until the walk has moved past the crest
or downstream at least as far as the crest is from the channel. After that
the channel can be rejoined. Gearon's Figure 4 tool instead raises the whole
parent channel by 1,000 m for the entire walk.

This module does not change the thalweg-slope lambda.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# (drow, dcol), same order as walk_funcs.precompute_neighbors.
_OFFSETS = np.array(
    [
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1],
        [-1, -1],
        [1, -1],
        [-1, 1],
        [1, 1],
    ],
    dtype=int,
)


def aggregation_factor(native_cell: float, cell_m: float) -> int:
    """How many native cells to average into one walker cell.

    ``cell_m`` of 0, or a cell already coarser than ``cell_m``, keeps the
    native grid.
    """
    if cell_m is None or float(cell_m) <= float(native_cell):
        return 1
    return max(1, int(round(float(cell_m) / float(native_cell))))


def aggregate_elevation(elevation: np.ndarray, factor: int) -> np.ndarray:
    """Average the DEM into ``factor`` by ``factor`` blocks.

    The walker scores slope as rise per pixel, as Gearon's code does on a
    ~30 m elevation model. On a 1 m grid that rise is about a millimetre
    along a river, so a furrow outranks the valley. Averaging to a coarser
    cell removes the finer relief and lets one step feel the slope over
    that distance. Nodata stays nodata. A partial block at the edge is dropped.
    """
    factor = int(factor)
    z = np.asarray(elevation, dtype=float)
    if factor <= 1:
        return z
    height = z.shape[0] // factor * factor
    width = z.shape[1] // factor * factor
    blocks = z[:height, :width].reshape(height // factor, factor, width // factor, factor)
    valid = np.isfinite(blocks)
    count = valid.sum(axis=(1, 3))
    total = np.where(valid, blocks, 0.0).sum(axis=(1, 3))
    out = np.full(count.shape, np.nan, dtype=float)
    np.divide(total, count, out=out, where=count > 0)
    return out


@dataclass
class RoutingPathSlope:
    """Slope summary for one walk origin."""

    slope: float
    n_frequent_cells: int
    n_frequent_visits: int
    modal_path_slope: float
    modal_path_length: float
    path_rows: np.ndarray
    path_cols: np.ndarray


def routing_slope_at_seed(
    elevation: np.ndarray,
    seed_row: int,
    seed_col: int,
    *,
    xres: float,
    yres: float,
    n_steps: int,
    n_trials: int,
    theta: float = 10.0,
    alpha: float = 1.0,
    beta: float = 0.5,
    frequency_threshold: float = 0.075,
    outward: tuple[float, float] | None = None,
    rng: np.random.Generator | None = None,
) -> RoutingPathSlope:
    """Run ``n_trials`` walks and return the slope on the frequent paths.

    ``elevation`` is a 2-D array in native DEM units. Nodata is NaN.
    ``xres`` and ``yres`` are positive cell sizes in working-CRS units.
    ``slope`` is positive when the frequent steps descend.
    ``outward`` is the pixel vector ``(drow, dcol)`` from the channel to the
    seed. When it is set, the walk cannot step back toward the channel until
    it is past the crest or downstream of the start.
    """
    z = np.asarray(elevation, dtype=float)
    if z.ndim != 2:
        raise ValueError("elevation must be a 2-D array.")
    if n_steps < 1 or n_trials < 1:
        raise ValueError("n_steps and n_trials must be at least 1.")
    if xres <= 0 or yres <= 0:
        raise ValueError("xres and yres must be positive.")
    nrows, ncols = z.shape
    if not (0 <= seed_row < nrows and 0 <= seed_col < ncols):
        return _empty_result()
    if not np.isfinite(z[seed_row, seed_col]):
        return _empty_result()

    rng = rng or np.random.default_rng()
    visit, slope_sum = accumulate_visit_frequency(
        z,
        int(seed_row),
        int(seed_col),
        xres=float(xres),
        yres=float(yres),
        n_steps=int(n_steps),
        n_trials=int(n_trials),
        theta=float(theta),
        alpha=float(alpha),
        beta=float(beta),
        outward=outward,
        rng=rng,
    )

    # Fraction of the busiest cell, matching the Figure 4 caption
    # ("normalized" occupation). A threshold on visits/trials keeps almost
    # every cell once the walk is confined to a narrow channel.
    peak = int(visit.max()) if visit.size else 0
    if peak <= 0:
        frequent = np.zeros(visit.shape, dtype=bool)
    else:
        frequent = visit >= float(frequency_threshold) * peak
    n_cells = int(frequent.sum())
    n_hits = int(visit[frequent].sum()) if n_cells else 0
    if n_hits > 0:
        slope = float(slope_sum[frequent].sum() / n_hits)
    else:
        slope = float("nan")

    path_rows, path_cols = _modal_path(
        visit,
        int(seed_row),
        int(seed_col),
        int(n_steps),
        outward=outward,
    )
    modal_slope, modal_length = _net_slope(z, path_rows, path_cols, float(xres), float(yres))
    return RoutingPathSlope(
        slope=slope,
        n_frequent_cells=n_cells,
        n_frequent_visits=n_hits,
        modal_path_slope=modal_slope,
        modal_path_length=modal_length,
        path_rows=path_rows,
        path_cols=path_cols,
    )


def accumulate_visit_frequency(
    elevation: np.ndarray,
    seed_row: int,
    seed_col: int,
    *,
    xres: float,
    yres: float,
    n_steps: int,
    n_trials: int,
    theta: float = 10.0,
    alpha: float = 1.0,
    beta: float = 0.5,
    outward: tuple[float, float] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Visit counts and summed map slopes from ``n_trials`` softmax walks."""
    z = np.asarray(elevation, dtype=float)
    rng = rng or np.random.default_rng()
    visit = np.zeros(z.shape, dtype=np.int32)
    slope_sum = np.zeros(z.shape, dtype=float)
    if not (0 <= seed_row < z.shape[0] and 0 <= seed_col < z.shape[1]):
        return visit, slope_sum
    if not np.isfinite(z[seed_row, seed_col]):
        return visit, slope_sum
    for _ in range(int(n_trials)):
        _walk_once(
            z,
            int(seed_row),
            int(seed_col),
            int(n_steps),
            float(theta),
            float(alpha),
            float(beta),
            float(xres),
            float(yres),
            visit,
            slope_sum,
            rng,
            outward=outward,
        )
    return visit, slope_sum


def _empty_result() -> RoutingPathSlope:
    return RoutingPathSlope(
        slope=float("nan"),
        n_frequent_cells=0,
        n_frequent_visits=0,
        modal_path_slope=float("nan"),
        modal_path_length=0.0,
        path_rows=np.empty(0, dtype=int),
        path_cols=np.empty(0, dtype=int),
    )


def _walk_once(
    z: np.ndarray,
    row: int,
    col: int,
    n_steps: int,
    theta: float,
    alpha: float,
    beta: float,
    xres: float,
    yres: float,
    visit: np.ndarray,
    slope_sum: np.ndarray,
    rng: np.random.Generator,
    outward: tuple[float, float] | None = None,
) -> None:
    """One softmax walk. Updates ``visit`` and ``slope_sum`` at arrival cells."""
    nrows, ncols = z.shape
    seed_row, seed_col = row, col
    last_row = last_col = -1
    last_vec = np.zeros(2, dtype=float)

    for _ in range(n_steps):
        cand_row = row + _OFFSETS[:, 0]
        cand_col = col + _OFFSETS[:, 1]
        inside = (
            (cand_row >= 0)
            & (cand_row < nrows)
            & (cand_col >= 0)
            & (cand_col < ncols)
        )
        not_prev = ~((cand_row == last_row) & (cand_col == last_col))
        keep = np.flatnonzero(inside & not_prev)
        if keep.size == 0:
            break

        cr = cand_row[keep]
        cc = cand_col[keep]
        dr = _OFFSETS[keep, 0].astype(float)
        dc = _OFFSETS[keep, 1].astype(float)
        dest_z = z[cr, cc]
        finite = np.isfinite(dest_z) & np.isfinite(z[row, col])
        if not np.any(finite):
            break
        cr = cr[finite]
        cc = cc[finite]
        dr = dr[finite]
        dc = dc[finite]
        rise = z[row, col] - z[cr, cc]
        leaving = _steps_leave_the_channel(row, col, seed_row, seed_col, dr, dc, outward)
        if not np.all(leaving) and np.any(leaving):
            cr = cr[leaving]
            cc = cc[leaving]
            dr = dr[leaving]
            dc = dc[leaving]
            rise = rise[leaving]
        pixel_dist = np.hypot(dr, dc)
        map_dist = np.hypot(dr * yres, dc * xres)
        pixel_slope = rise / pixel_dist
        map_slope = rise / map_dist

        direction = np.column_stack([dr, dc]) / pixel_dist[:, None]
        last_norm = float(np.linalg.norm(last_vec))
        if last_norm > 0:
            cosine = direction @ (last_vec / last_norm)
        else:
            cosine = np.zeros(cr.shape[0], dtype=float)
        momentum = alpha * (1.0 + cosine) / 2.0
        weights = (1.0 - beta) * momentum + beta * (theta * pixel_slope)
        probabilities = _softmax(weights)
        choice = int(rng.choice(cr.shape[0], p=probabilities))

        new_row = int(cr[choice])
        new_col = int(cc[choice])
        visit[new_row, new_col] += 1
        slope_sum[new_row, new_col] += float(map_slope[choice])
        last_vec = np.array([new_row - row, new_col - col], dtype=float)
        last_row, last_col = row, col
        row, col = new_row, new_col


def _softmax(weights: np.ndarray) -> np.ndarray:
    shifted = weights - np.max(weights)
    raw = np.exp(shifted)
    total = float(raw.sum())
    if not np.isfinite(total) or total <= 0:
        return np.full(weights.shape, 1.0 / weights.size)
    return raw / total


def _steps_leave_the_channel(
    row: int,
    col: int,
    seed_row: int,
    seed_col: int,
    dr: np.ndarray,
    dc: np.ndarray,
    outward: tuple[float, float] | None,
) -> np.ndarray:
    """Steps that are allowed while the walk is still at the starting bank.

    ``outward`` points from the channel to the seed, in pixel ``(drow, dcol)``.
    Its length is how far the crest is from the channel. Until the walker is
    past that crest, or at least that far downstream along the bank, a step
    back toward the channel is refused. After either of those, every step is
    allowed, including a later return to the channel.
    """
    allowed = np.ones(dr.shape[0], dtype=bool)
    if outward is None:
        return allowed
    oy, ox = float(outward[0]), float(outward[1])
    span = float(np.hypot(oy, ox))
    if span < 1e-6:
        return allowed
    uy, ux = oy / span, ox / span
    rel_y = float(row - seed_row)
    rel_x = float(col - seed_col)
    outward_progress = rel_y * uy + rel_x * ux
    downstream = abs(rel_y * (-ux) + rel_x * uy)
    if outward_progress > 0.5 or downstream > span:
        return allowed
    return (dr * uy + dc * ux) >= -1e-8


def _modal_path(
    visit: np.ndarray,
    seed_row: int,
    seed_col: int,
    n_steps: int,
    outward: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy trace through the cells visited most often."""
    nrows, ncols = visit.shape
    row, col = seed_row, seed_col
    last_row = last_col = -1
    rows = [row]
    cols = [col]
    for _ in range(n_steps):
        cand_row = row + _OFFSETS[:, 0]
        cand_col = col + _OFFSETS[:, 1]
        inside = (
            (cand_row >= 0)
            & (cand_row < nrows)
            & (cand_col >= 0)
            & (cand_col < ncols)
        )
        not_prev = ~((cand_row == last_row) & (cand_col == last_col))
        keep = np.flatnonzero(inside & not_prev)
        if keep.size == 0:
            break
        cr = cand_row[keep]
        cc = cand_col[keep]
        dr = _OFFSETS[keep, 0].astype(float)
        dc = _OFFSETS[keep, 1].astype(float)
        counts = visit[cr, cc].astype(float)
        leaving = _steps_leave_the_channel(row, col, seed_row, seed_col, dr, dc, outward)
        if np.any(leaving):
            counts = np.where(leaving, counts, -1.0)
        best = float(np.max(counts))
        if best <= 0:
            break
        choice = int(np.argmax(counts))
        new_row = int(cr[choice])
        new_col = int(cc[choice])
        rows.append(new_row)
        cols.append(new_col)
        last_row, last_col = row, col
        row, col = new_row, new_col
    return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)


def _net_slope(
    z: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    xres: float,
    yres: float,
) -> tuple[float, float]:
    if rows.size < 2:
        return float("nan"), 0.0
    length = 0.0
    for i in range(1, rows.size):
        length += float(
            np.hypot((rows[i] - rows[i - 1]) * yres, (cols[i] - cols[i - 1]) * xres)
        )
    if length <= 0 or not np.isfinite(z[rows[0], cols[0]]) or not np.isfinite(z[rows[-1], cols[-1]]):
        return float("nan"), length
    return float((z[rows[0], cols[0]] - z[rows[-1], cols[-1]]) / length), length
