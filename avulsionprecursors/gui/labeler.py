"""Cross-section labeling interface."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from rasterio.warp import transform_bounds
from scipy.signal import savgol_filter

from .config import GUIConfig


class CrossSectionLabeler:
    """Interactive GUI for labeling river cross-sections."""

    def __init__(
        self,
        config: GUIConfig,
        output_dir: Path,
        river_name: str,
        dem_path: Optional[Path] = None,
        show_satellite: bool = True,
        show_dem: bool = True,
        dem_alpha: float = 0.6,
        satellite_alpha: float = 0.7,
        node_id: Optional[int] = None,
    ):
        self.config = config
        self.output_dir = output_dir
        self.river_name = river_name
        self.node_id = node_id
        self.dem_path = Path(dem_path) if dem_path else None
        self.show_satellite = show_satellite
        self.show_dem = show_dem
        self.dem_alpha = dem_alpha
        self.satellite_alpha = satellite_alpha
        self.current_label_idx = 0
        self.plotted_points: List[Any] = []
        self.labeled_points: Dict[str, List[Tuple[float, float]]] = {
            label: [] for label in config.labels
        }
        self.panning = False
        self.pan_start: Optional[Tuple[float, float]] = None
        self.zooming = False
        self.zoom_rect: Optional[Rectangle] = None
        # Current profile for optional snap-to-smooth (x ascending)
        self._profile_x: Optional[np.ndarray] = None
        self._profile_z_smooth: Optional[np.ndarray] = None

    def setup_figure(self) -> Tuple[Figure, List[Axes]]:
        """Create figure with a wide profile on top and DEM/satellite maps below."""
        fig = plt.figure(figsize=(22, 12))
        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[1.6, 1.0],
            width_ratios=[1.0, 1.0],
            hspace=0.28,
            wspace=0.18,
            left=0.05,
            right=0.98,
            top=0.94,
            bottom=0.06,
        )
        ax1 = fig.add_subplot(gs[0, :])  # full-width stretched profile
        ax2 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
        ax3 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
        return fig, [ax1, ax2, ax3]

    def setup_event_handlers(self, fig: Figure, axes: List[Axes]) -> None:
        self.fig = fig
        self.ax1 = axes[0]
        self.ax2 = axes[1]
        self.ax3 = axes[2] if len(axes) > 2 else None

        self.cid_click = fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.cid_move = fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.cid_release = fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.cid_scroll = fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.cid_key = fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_click(self, event: Any) -> None:
        if event.inaxes != self.ax1:
            return

        if event.button == 2:
            self.panning = True
            self.pan_start = (event.xdata, event.ydata)
            return

        if self.panning or self.zooming:
            return

        if self.current_label_idx >= len(self.config.labels):
            return

        current_label = self.config.labels[self.current_label_idx]
        x = float(event.xdata)
        y = float(event.ydata)
        # Snap elevation to smoothed profile at clicked distance (keeps pick stable)
        if (
            self._profile_x is not None
            and self._profile_z_smooth is not None
            and len(self._profile_x) >= 2
        ):
            y = float(np.interp(x, self._profile_x, self._profile_z_smooth))
        self.labeled_points[current_label].append((x, y))

        point = self.ax1.plot(
            x,
            y,
            "X",
            markersize=10,
            color=self.config.colors.get(current_label, "tab:red"),
        )[0]
        self.plotted_points.append(point)

        self.current_label_idx += 1
        self._update_title()
        self.fig.canvas.draw()

    def _on_move(self, event: Any) -> None:
        if self.panning and event.inaxes == self.ax1:
            if self.pan_start is None:
                return
            dx = self.pan_start[0] - event.xdata
            dy = self.pan_start[1] - event.ydata
            self.ax1.set_xlim(self.ax1.get_xlim() + dx)
            self.ax1.set_ylim(self.ax1.get_ylim() + dy)
            self.fig.canvas.draw_idle()

    def _on_release(self, event: Any) -> None:
        if event.button == 2:
            self.panning = False
            self.pan_start = None

    def _on_scroll(self, event: Any) -> None:
        if event.inaxes != self.ax1:
            return

        cur_xlim = self.ax1.get_xlim()
        cur_ylim = self.ax1.get_ylim()

        xdata, ydata = event.xdata, event.ydata
        base_scale = 1.1
        scale_factor = 1 / base_scale if event.button == "up" else base_scale

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        self.ax1.set_xlim([
            xdata - new_width * (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0]),
            xdata + new_width * (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0]),
        ])
        self.ax1.set_ylim([
            ydata - new_height * (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0]),
            ydata + new_height * (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0]),
        ])

        self.fig.canvas.draw_idle()

    @staticmethod
    def _smooth_profile(
        dist: np.ndarray,
        elev: np.ndarray,
        window_m: float,
        polyorder: int,
    ) -> np.ndarray:
        """Savitzky–Golay smooth along-track; returns elev-sized array (NaN preserved)."""
        z = elev.astype(float).copy()
        valid = np.isfinite(dist) & np.isfinite(z)
        if valid.sum() < 5:
            return z

        x = dist[valid]
        y = z[valid]
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        dx = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
        if not np.isfinite(dx) or dx <= 0:
            dx = 1.0
        win = int(round(float(window_m) / dx))
        if win % 2 == 0:
            win += 1
        win = int(np.clip(win, 5, len(y) - (1 - len(y) % 2)))
        if win % 2 == 0:
            win -= 1
        win = max(5, min(win, len(y) if len(y) % 2 == 1 else len(y) - 1))
        if win < 5 or win > len(y):
            return z
        poly = min(int(polyorder), win - 1)
        if poly < 1:
            return z

        y_s = savgol_filter(y, window_length=win, polyorder=poly, mode="interp")
        # map back to original index order
        z_out = z.copy()
        valid_idx = np.flatnonzero(valid)[order]
        z_out[valid_idx] = y_s
        return z_out

    def plot_cross_section(
        self,
        ax: plt.Axes,
        df: pd.DataFrame,
        predicted_points: Optional[Dict[str, float]] = None,
    ) -> None:
        if "elevation" not in df.columns:
            df["elevation"] = float("nan")

        x = df["dist_along"].to_numpy(dtype=float)
        z = df["elevation"].to_numpy(dtype=float)
        ax.plot(
            x,
            z,
            "-o",
            markersize=2,
            color="0.55",
            lw=0.8,
            label="raw profile",
            zorder=2,
        )

        self._profile_x = None
        self._profile_z_smooth = None
        if getattr(self.config, "show_smoothed_profile", True):
            z_s = self._smooth_profile(
                x,
                z,
                window_m=float(getattr(self.config, "smooth_window_m", 25.0)),
                polyorder=int(getattr(self.config, "smooth_polyorder", 2)),
            )
            ax.plot(
                x,
                z_s,
                "-",
                color="tab:blue",
                lw=2.0,
                label=f"smoothed (~{getattr(self.config, 'smooth_window_m', 25):.0f} m)",
                zorder=3,
            )
            # Store sorted copies for click snap
            valid = np.isfinite(x) & np.isfinite(z_s)
            if valid.sum() >= 2:
                order = np.argsort(x[valid])
                self._profile_x = x[valid][order]
                self._profile_z_smooth = z_s[valid][order]

        ax.set_xlabel("Along Track Distance")
        ax.set_ylabel("Elevation")
        ax.legend(loc="best", fontsize=8)
        # Prefer a low, wide data box so along-track detail is stretched
        try:
            ax.set_box_aspect(0.28)
        except Exception:
            pass

        if predicted_points:
            for label, dist in predicted_points.items():
                feature_name = label.split("_")[0]
                y = np.interp(dist, df["dist_along"], df["elevation"])
                ax.plot(
                    dist,
                    y,
                    "X",
                    markersize=10,
                    color=self.config.colors.get(feature_name, "tab:red"),
                    alpha=0.5,
                )

    def plot_map(
        self,
        ax: plt.Axes,
        df: gpd.GeoDataFrame,
        map_type: str = "both",
        zoom_level: int = 15,
    ) -> None:
        if df.crs and df.crs.to_epsg() != 4326:
            df_plot = df.to_crs(epsg=4326)
            original_crs = df.crs
        else:
            df_plot = df
            original_crs = df.crs if df.crs else None

        if df_plot.empty or df_plot.geometry.isnull().all():
            print("⚠️  Warning: No valid geometry for map display")
            return

        valid_mask = ~df_plot.geometry.isnull()
        if not valid_mask.any():
            print("⚠️  Warning: No valid geometries for map display")
            return
        df_plot = df_plot[valid_mask]

        bounds_wgs84 = df_plot.total_bounds
        if not all(np.isfinite(bounds_wgs84)):
            print(f"⚠️  Warning: Invalid bounds {bounds_wgs84}, using default extent")
            lon_center = np.nanmean(df_plot.geometry.x) if valid_mask.any() else 0
            lat_center = np.nanmean(df_plot.geometry.y) if valid_mask.any() else 0
            if not (np.isfinite(lon_center) and np.isfinite(lat_center)):
                print("⚠️  Warning: Cannot determine map center, skipping map display")
                return
            bounds_wgs84 = [lon_center - 0.01, lat_center - 0.01, lon_center + 0.01, lat_center + 0.01]

        buffer = 0.002  # degrees
        extent_wgs84 = [
            bounds_wgs84[0] - buffer,
            bounds_wgs84[2] + buffer,
            bounds_wgs84[1] - buffer,
            bounds_wgs84[3] + buffer,
        ]

        if map_type in ("dem", "both"):
            if self.show_dem and self.dem_path and self.dem_path.exists():
                try:
                    self._plot_dem_layer(ax, extent_wgs84, original_crs, alpha=1.0)
                except Exception as e:
                    print(f"⚠️  Warning: Could not load DEM: {e}")
                    warnings.warn(f"DEM display failed: {e}")

        if map_type in ("satellite", "both"):
            if self.show_satellite:
                try:
                    osm_background = cimgt.GoogleTiles(style="satellite", cache=True)
                    ax.add_image(osm_background, zoom_level, interpolation="spline36", alpha=1.0)
                except Exception as e:
                    print(f"⚠️  Warning: Could not load satellite imagery: {e}")

        if len(df_plot) > 1:
            x_coords = [p.x for p in df_plot.geometry]
            y_coords = [p.y for p in df_plot.geometry]
            ax.plot(
                x_coords,
                y_coords,
                "k-",
                linewidth=2,
                transform=ccrs.PlateCarree(),
                label="Cross-section",
                zorder=5,
            )

        elevation_values = df_plot["elevation"].values if "elevation" in df_plot.columns else None
        if elevation_values is not None:
            valid_elev = np.isfinite(elevation_values)
            if valid_elev.any():
                df_valid = df_plot[valid_elev]
                scatter = ax.scatter(
                    df_valid.geometry.x,
                    df_valid.geometry.y,
                    c=elevation_values[valid_elev],
                    cmap="terrain",
                    marker="o",
                    edgecolor="k",
                    linewidth=0.8,
                    s=15,
                    transform=ccrs.PlateCarree(),
                    zorder=10,
                )
                if map_type == "dem":
                    plt.colorbar(scatter, ax=ax, orientation="vertical", label="Elevation (m)", shrink=0.8)
            else:
                ax.scatter(
                    df_plot.geometry.x,
                    df_plot.geometry.y,
                    color="red",
                    marker="o",
                    edgecolor="k",
                    linewidth=0.8,
                    s=15,
                    transform=ccrs.PlateCarree(),
                    zorder=10,
                )
        else:
            ax.scatter(
                df_plot.geometry.x,
                df_plot.geometry.y,
                color="red",
                marker="o",
                edgecolor="k",
                linewidth=0.8,
                s=15,
                transform=ccrs.PlateCarree(),
                zorder=10,
            )

        try:
            ax.gridlines(draw_labels=True, alpha=0.5, linestyle="--")
        except Exception:
            pass

        if hasattr(ax, "set_extent"):
            try:
                ax.set_extent(extent_wgs84, crs=ccrs.PlateCarree())
            except Exception as e:
                print(f"⚠️  Warning: Could not set map extent: {e}")
        else:
            ax.set_xlim(extent_wgs84[0], extent_wgs84[1])
            ax.set_ylim(extent_wgs84[2], extent_wgs84[3])

    def _plot_dem_layer(
        self,
        ax: plt.Axes,
        extent_wgs84: List[float],
        original_crs: Any = None,
        alpha: float = 1.0,
    ) -> None:
        with rasterio.open(self.dem_path) as dem_src:
            dem_bounds = dem_src.bounds
            dem_crs = dem_src.crs

            try:
                bounds_dem_crs = transform_bounds(
                    "EPSG:4326",
                    str(dem_crs),
                    extent_wgs84[0],
                    extent_wgs84[2],
                    extent_wgs84[1],
                    extent_wgs84[3],
                )
            except Exception as e:
                print(f"⚠️  Warning: Could not transform bounds: {e}")
                return

            west = max(bounds_dem_crs[0], dem_bounds.left)
            east = min(bounds_dem_crs[2], dem_bounds.right)
            south = max(bounds_dem_crs[1], dem_bounds.bottom)
            north = min(bounds_dem_crs[3], dem_bounds.top)

            if west >= east or south >= north:
                print("⚠️  Warning: Extent outside DEM bounds")
                return

            window = rasterio.windows.from_bounds(west, south, east, north, dem_src.transform)

            try:
                dem_data = dem_src.read(1, window=window, masked=True)
            except Exception as e:
                print(f"⚠️  Warning: Could not read DEM data: {e}")
                return

            height, width = dem_data.shape
            x = np.linspace(west, east, width)
            y = np.linspace(north, south, height)
            X, Y = np.meshgrid(x, y)

            max_pixels = 500
            if height * width > max_pixels * max_pixels:
                downsample_factor = int(np.ceil(np.sqrt(height * width / (max_pixels * max_pixels))))
                dem_data = dem_data[::downsample_factor, ::downsample_factor]
                X = X[::downsample_factor, ::downsample_factor]
                Y = Y[::downsample_factor, ::downsample_factor]

            x_flat = X.flatten()
            y_flat = Y.flatten()
            lon_flat, lat_flat = rasterio.warp.transform(
                str(dem_crs), "EPSG:4326", x_flat, y_flat
            )
            lon_coords = np.array(lon_flat).reshape(X.shape)
            lat_coords = np.array(lat_flat).reshape(Y.shape)

            dem_data = np.ma.masked_invalid(dem_data)

            try:
                ax.pcolormesh(
                    lon_coords,
                    lat_coords,
                    dem_data,
                    cmap="terrain",
                    alpha=alpha,
                    transform=ccrs.PlateCarree(),
                    shading="auto",
                    vmin=np.nanpercentile(dem_data, 2),
                    vmax=np.nanpercentile(dem_data, 98),
                )
            except Exception as e:
                print(f"⚠️  Warning: Could not plot DEM: {e}")
                return

    def _on_key(self, event: Any) -> None:
        if event.key == "u":
            self._undo_last_point()
        elif event.key == "d":
            self._save_and_close()
        elif event.key == "r":
            self._reset_view()
        elif event.key == "h":
            self._show_help()

    def _xs_prefix(self) -> str:
        if self.node_id is not None:
            return f"Cross-section {int(self.node_id)}"
        return self.river_name

    def _set_window_title(self) -> None:
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is None:
            return
        title = self._xs_prefix()
        try:
            manager.set_window_title(title)
        except Exception:
            pass

    def _update_title(self) -> None:
        prefix = self._xs_prefix()
        if self.current_label_idx >= len(self.config.labels):
            title = f"{prefix}  —  All points labeled. Press 'd' to save and continue."
        else:
            next_label = self.config.labels[self.current_label_idx]
            title = f"{prefix}  —  Pick {next_label} point"
        self.ax1.set_title(title, fontsize=14, fontweight="bold")

    def _undo_last_point(self) -> None:
        if self.current_label_idx > 0 and self.plotted_points:
            self.current_label_idx -= 1
            current_label = self.config.labels[self.current_label_idx]
            if self.labeled_points[current_label]:
                self.labeled_points[current_label].pop()
            if self.plotted_points:
                point = self.plotted_points.pop()
                point.remove()
            self._update_title()
            self.fig.canvas.draw()

    def _save_and_close(self) -> None:
        self._save_labels()
        plt.close(self.fig)

    def _reset_view(self) -> None:
        self.ax1.autoscale()
        self.fig.canvas.draw()

    def _show_help(self) -> None:
        help_text = """
        Controls:
        - Left click: Place point (elevation snaps to smoothed profile)
        - Middle click + drag: Pan
        - Scroll wheel: Zoom
        - 'u': Undo last point
        - 'd': Done/Save
        - 'r': Reset view
        - 'h': Show this help

        Profile panel shows raw DEM (gray) and Savitzky–Golay smooth (blue).
        """
        print(help_text)

    def _save_labels(self) -> None:
        output_file = self.output_dir / f"{self.river_name}_labels.csv"
        records = []
        for label, points in self.labeled_points.items():
            if points:
                x, y = points[0]
                records.append({"label": label, "dist_along": x, "elevation": y})
        new_df = pd.DataFrame(records)
        if output_file.exists() and output_file.stat().st_size > 0:
            try:
                old = pd.read_csv(output_file, comment="#")
            except Exception:
                old = pd.DataFrame()
            if not old.empty and "label" in old.columns:
                replaced = set(new_df["label"]) if not new_df.empty else set()
                keep = old[~old["label"].astype(str).isin(replaced)]
                new_df = pd.concat([keep, new_df], ignore_index=True)
                order = {
                    lab: i
                    for i, lab in enumerate(
                        ["channel", "ridge1", "floodplain1", "ridge2", "floodplain2"]
                    )
                }
                new_df["_ord"] = new_df["label"].map(lambda x: order.get(str(x), 99))
                new_df = new_df.sort_values("_ord").drop(columns="_ord")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        new_df.to_csv(output_file, index=False)

    def label_cross_section(
        self,
        df: gpd.GeoDataFrame,
        predicted_points: Optional[Dict[str, float]] = None,
    ) -> Dict[str, List[Tuple[float, float]]]:
        fig, axes = self.setup_figure()
        self.setup_event_handlers(fig, axes)

        self.plot_cross_section(axes[0], df, predicted_points)

        if self.show_dem and self.dem_path and self.dem_path.exists():
            try:
                self.plot_map(axes[1], df, map_type="dem")
                axes[1].set_title("DEM Terrain", fontsize=12, fontweight="bold")
            except Exception as e:
                print(f"⚠️  Error plotting DEM: {e}")
                axes[1].text(
                    0.5,
                    0.5,
                    f"DEM Error:\n{str(e)[:50]}",
                    ha="center",
                    va="center",
                    transform=axes[1].transAxes,
                    fontsize=10,
                    color="red",
                )
                axes[1].set_title("DEM Terrain (Error)", fontsize=12)
        else:
            msg = f"DEM not found:\n{self.dem_path}" if self.dem_path else "DEM not configured\n(use --dem-path)"
            axes[1].text(0.5, 0.5, msg, ha="center", va="center", transform=axes[1].transAxes, fontsize=10)
            axes[1].set_title("DEM Terrain (Not Available)", fontsize=12)

        if self.show_satellite:
            self.plot_map(axes[2], df, map_type="satellite")
            axes[2].set_title("Satellite Imagery", fontsize=12, fontweight="bold")
        else:
            axes[2].text(0.5, 0.5, "Satellite not available", ha="center", va="center", transform=axes[2].transAxes)
            axes[2].set_title("Satellite Imagery (Not Available)", fontsize=12)

        self._set_window_title()
        self._update_title()
        self._raise_window()

        plt.show()
        return self.labeled_points

    def _raise_window(self) -> None:
        """Bring the matplotlib window to the front (macOS often hides it)."""
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None:
            try:
                manager.show()
            except Exception:
                pass
            win = getattr(manager, "window", None)
            if win is not None:
                for meth in ("raise_", "makeKeyAndOrderFront_"):
                    fn = getattr(win, meth, None)
                    if callable(fn):
                        try:
                            fn()
                        except Exception:
                            try:
                                fn(win)
                            except Exception:
                                pass
        try:
            from AppKit import NSApplicationActivateIgnoringOtherApps, NSRunningApplication
            import os

            app = NSRunningApplication.runningApplicationWithProcessIdentifier_(os.getpid())
            if app is not None:
                app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
        except Exception:
            pass
