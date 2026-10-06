"""Input checks with plain-language error messages."""

from __future__ import annotations

from pathlib import Path

from pyproj import CRS

from avulsionprecursors.config import WorkflowConfig
from avulsionprecursors.exceptions import CRSError, InputValidationError, UnitsError
from avulsionprecursors.io.raster import SUPPORTED_RASTER_SUFFIXES

POSITIVE_EXTRACT_FIELDS = (
    "spacing",
    "cross_half_length",
    "max_half_length",
    "valley_wall_dz",
    "valley_wall_persist",
    "channel_elev_window",
    "backwater_length",
    "slope_smooth_window",
    "slope_sample_step",
    "thalweg_half_width",
)


def parse_crs(value: str, label: str = "CRS") -> CRS:
    try:
        return CRS.from_user_input(value)
    except Exception as exc:
        raise CRSError(
            f"Could not understand {label} {value!r}.\n"
            "Use 'UTM' to choose a UTM zone automatically, or an EPSG code "
            "such as EPSG:32610.\n"
            f"Details: {exc}"
        ) from exc


def crs_is_geographic(crs: CRS) -> bool:
    try:
        return bool(crs.is_geographic)
    except Exception:
        return False


def linear_unit_name(crs: CRS) -> str:
    try:
        return (crs.axis_info[0].unit_name or "").lower()
    except Exception:
        return ""


def require_projected(crs: CRS, label: str) -> None:
    if crs_is_geographic(crs):
        raise CRSError(
            f"{label} is geographic (longitude/latitude, degrees).\n"
            "Cross-section distances need a projected CRS in metres or feet "
            "(for example UTM). Set crs.target to 'UTM' or an EPSG code such "
            "as EPSG:32610."
        )
    unit = linear_unit_name(crs)
    if unit and any(token in unit for token in ("degree", "radian")):
        raise UnitsError(
            f"{label} uses angular units ({unit}). "
            "Choose a projected working CRS (UTM or another EPSG in metres or feet)."
        )


def ensure_writable_directory(path: Path) -> None:
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        raise InputValidationError(
            f"Cannot write to the output folder:\n  {path}\n"
            "Choose a folder you have permission to write, or create it first.\n"
            f"Details: {exc}"
        ) from exc


def validate_config(cfg: WorkflowConfig) -> None:
    """Raise InputValidationError if required inputs or parameters are invalid."""
    dem = Path(cfg.paths.dem)
    centerline = Path(cfg.paths.centerline)

    if not dem.exists():
        raise InputValidationError(
            f"Could not find the DEM file:\n  {dem}\n"
            "Set paths.dem in the config, or pass --dem. "
            "Relative paths are resolved from the config file's folder."
        )
    if dem.suffix.lower() not in SUPPORTED_RASTER_SUFFIXES:
        raise InputValidationError(
            f"Unsupported DEM file type {dem.suffix!r}:\n  {dem}\n"
            "Use a GDAL-readable raster such as GeoTIFF (.tif), .vrt, or .img."
        )
    if not centerline.exists():
        raise InputValidationError(
            f"Could not find the centerline file:\n  {centerline}\n"
            "Set paths.centerline to a line shapefile or GeoPackage."
        )

    for optional, label in (
        (cfg.paths.channel_polygon, "channel polygon"),
        (cfg.paths.centerline_points, "centerline points"),
        (cfg.paths.clip_boundary, "clip boundary"),
        (cfg.paths.levee, "levee overlay"),
    ):
        if optional is not None and not Path(optional).exists():
            raise InputValidationError(
                f"Could not find the {label} file:\n  {optional}\n"
                f"If you do not want to use a {label}, set it to null in the config."
            )

    ext = cfg.extract
    for name in POSITIVE_EXTRACT_FIELDS:
        value = getattr(ext, name)
        if value is None or float(value) <= 0:
            raise InputValidationError(
                f"extract.{name} must be a positive number (got {value!r})."
            )
    if ext.max_half_length < ext.cross_half_length:
        raise InputValidationError(
            "extract.max_half_length must be greater than or equal to "
            f"extract.cross_half_length ({ext.max_half_length} < {ext.cross_half_length})."
        )
    if ext.sample_step is not None and ext.sample_step <= 0:
        raise InputValidationError("extract.sample_step must be positive if set.")
    if ext.slope_min <= 0:
        raise InputValidationError("extract.slope_min must be positive.")
    if ext.slope_smooth_window < ext.slope_sample_step:
        raise InputValidationError(
            "extract.slope_smooth_window must be greater than extract.slope_sample_step."
        )
    if ext.max_elev_above_channel is not None and ext.max_elev_above_channel <= 0:
        raise InputValidationError(
            "extract.max_elev_above_channel must be positive, or null to disable."
        )

    target = str(cfg.crs.target).strip()
    if not target:
        raise CRSError("crs.target is empty. Use 'UTM' or an EPSG code such as EPSG:32610.")
    if target.upper() != "UTM":
        crs = parse_crs(target, "crs.target")
        require_projected(crs, "crs.target")

    ensure_writable_directory(cfg.output_dir())
