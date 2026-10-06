"""Load and document workflow settings from YAML, with optional CLI overrides.

All distance thresholds (spacing, transect length, smoothing windows) are in
the **horizontal units of the working CRS** after reprojection (usually metres).

All elevation thresholds (valley-wall rise, height-above-channel mask) are in
the **native vertical units of the DEM**. This package does not convert
elevation values from feet to metres or vice versa.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import yaml

from avulsionprecursors.exceptions import InputValidationError, UnitsError

ALLOWED_VERTICAL_UNITS = ("native", "m", "meter", "metre", "metres", "ft", "feet", "us_ft", "us-ft")
VERTICAL_UNIT_ALIASES = {
    "native": "native",
    "m": "m",
    "meter": "m",
    "metre": "m",
    "metres": "m",
    "ft": "ft",
    "feet": "ft",
    "us_ft": "us_ft",
    "us-ft": "us_ft",
}


def _as_path(value: Any) -> Optional[Path]:
    if value is None or value == "":
        return None
    return Path(str(value))


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


@dataclass
class PathsConfig:
    """Input and output locations. Relative paths are resolved from the config file directory."""

    dem: Path
    centerline: Path
    channel_polygon: Optional[Path] = None
    centerline_points: Optional[Path] = None
    clip_boundary: Optional[Path] = None
    levee: Optional[Path] = None  # optional line overlay of mapped artificial levees
    output_dir: Path = Path("outputs")


@dataclass
class CRSConfig:
    """Working coordinate system.

    ``target``:
        ``"UTM"`` to pick the UTM zone from the data centroid, or an EPSG
        string such as ``"EPSG:32610"``. The working CRS must be projected
        (metres or feet), not longitude/latitude.
    ``dem_source`` / ``centerline_source`` / ``vector_source``:
        Optional EPSG used only when a file is missing CRS metadata.
    """

    target: str = "UTM"
    dem_source: Optional[str] = None
    centerline_source: Optional[str] = None
    vector_source: Optional[str] = None


@dataclass
class UnitsConfig:
    """How to *label* vertical values. Elevations are never converted.

    Allowed: ``native``, ``m``, ``ft``, ``us_ft``.
    Use the same convention when you set elevation thresholds.
    """

    vertical: str = "native"


@dataclass
class ExtractConfig:
    """Cross-section extraction parameters.

    Horizontal values: working CRS units (typically metres after reprojection).
    Vertical values: native DEM units (not converted).
    """

    spacing: float = 200.0
    cross_half_length: float = 800.0
    max_half_length: float = 4000.0
    sample_step: Optional[float] = None  # default: DEM cell size
    valley_wall_dz: float = 30.0
    valley_wall_persist: float = 50.0
    valley_wall_skip_channel: float = 200.0
    valley_wall_keep: float = 100.0
    max_elev_above_channel: Optional[float] = 40.0
    channel_elev_window: float = 50.0
    backwater_length: float = 5000.0  # unused for Sm; kept for YAML compatibility
    centerline_smooth_window: float = 400.0
    centerline_smooth_sample: float = 10.0
    flow_tangent_half_delta: float = 100.0
    slope_min: float = 1e-5  # flag only; Sm is not floored
    default_slope: float = 1e-4  # unused; missing Sm stays NaN
    slope_smooth_window: float = 15000.0
    slope_sample_step: float = 5.0
    thalweg_half_width: float = 80.0


@dataclass
class LambdaConfig:
    """Lambda calculation options.

    The only ridge rule is the Gearon et al. lower-crest rule. Optional
    ``auto_minmax`` picks channel = profile minimum and ridges = left/right
    maxima. That can select valley walls when they are taller than levees, and
    it does not pick floodplains (those still come from label CSVs).
    """

    labels_dir: Optional[Path] = None
    auto_minmax: bool = False
    labels_since: Optional[str] = None


@dataclass
class LabelerConfig:
    """Interactive GUI options."""

    show_satellite: bool = True
    show_dem: bool = True
    skip_existing: bool = True
    overlay_existing: bool = True  # faded previous picks in the GUI
    levee_buffer: float = 25.0  # metres; profile ticks where the XS nears a mapped levee


@dataclass
class WorkflowConfig:
    """Full workflow configuration."""

    study_name: str = "study"
    paths: PathsConfig = field(
        default_factory=lambda: PathsConfig(dem=Path("dem.tif"), centerline=Path("centerline.shp"))
    )
    crs: CRSConfig = field(default_factory=CRSConfig)
    units: UnitsConfig = field(default_factory=UnitsConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    lambda_calc: LambdaConfig = field(default_factory=LambdaConfig)
    labeler: LabelerConfig = field(default_factory=LabelerConfig)
    config_path: Optional[Path] = None

    @property
    def vertical_unit(self) -> str:
        return VERTICAL_UNIT_ALIASES[self.units.vertical.lower()]

    def output_dir(self) -> Path:
        return Path(self.paths.output_dir)

    def prepared_dir(self) -> Path:
        return self.output_dir() / "prepared"

    def profiles_dir(self) -> Path:
        return self.output_dir() / "profiles"

    def labels_dir(self) -> Path:
        if self.lambda_calc.labels_dir is not None:
            return Path(self.lambda_calc.labels_dir)
        return self.output_dir() / "labels"

    def cross_sections_path(self) -> Path:
        return self.output_dir() / "cross_sections.shp"

    def centerline_points_path(self) -> Path:
        return self.output_dir() / "centerline_points.shp"

    def smoothed_centerline_path(self) -> Path:
        return self.output_dir() / "centerline_smoothed_for_azimuth.shp"

    def lambda_csv_path(self) -> Path:
        return self.output_dir() / f"{self.study_name}_lambda_results.csv"

    def lambda_plot_path(self) -> Path:
        return self.output_dir() / f"{self.study_name}_lambda_plot.png"

    def prepared_dem_path(self) -> Path:
        return self.prepared_dir() / "dem.tif"

    def prepared_centerline_path(self) -> Path:
        return self.prepared_dir() / "centerline.shp"

    def prepared_channel_polygon_path(self) -> Path:
        return self.prepared_dir() / "channel_polygon.shp"

    def prepared_clip_path(self) -> Path:
        return self.prepared_dir() / "clip_boundary.shp"


def normalize_vertical_units(value: str) -> str:
    key = str(value).strip().lower()
    if key not in VERTICAL_UNIT_ALIASES:
        allowed = "native, m, ft, us_ft"
        raise UnitsError(
            f"Unsupported vertical unit {value!r}.\n"
            f"Use one of: {allowed}.\n"
            "This setting only documents the DEM's stored elevation units. "
            "The workflow does not convert feet to metres. Set elevation "
            "thresholds (valley_wall_dz, max_elev_above_channel) in the same units."
        )
    return VERTICAL_UNIT_ALIASES[key]


def _update_dataclass(obj: Any, data: Mapping[str, Any]) -> None:
    if not data:
        return
    known = {f.name for f in fields(obj)}
    unknown = [k for k in data if k not in known]
    if unknown:
        raise InputValidationError(
            f"Unknown setting(s) {unknown} for {type(obj).__name__}. "
            f"Valid names: {sorted(known)}."
        )
    for key, value in data.items():
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, Mapping):
            _update_dataclass(current, value)
        else:
            setattr(obj, key, value)


def _resolve_path(path: Optional[Path], base: Path) -> Optional[Path]:
    if path is None:
        return None
    path = Path(path)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def load_config(path: Union[str, Path]) -> WorkflowConfig:
    """Read a YAML config file and resolve relative paths from that file's folder."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise InputValidationError(
            f"Could not find the configuration file:\n  {config_path}\n"
            "Pass a YAML file with --config, or see config/example.yaml."
        )
    try:
        raw = yaml.safe_load(config_path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise InputValidationError(
            f"The configuration file is not valid YAML:\n  {config_path}\n{exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise InputValidationError("The configuration file must contain a mapping of settings.")

    cfg = WorkflowConfig()
    cfg.config_path = config_path

    # Allow either lambda_calc or lambda in YAML
    if "lambda" in raw and "lambda_calc" not in raw:
        raw["lambda_calc"] = raw.pop("lambda")

    top_fields = {f.name for f in fields(cfg)}
    unknown = [k for k in raw if k not in top_fields]
    if unknown:
        raise InputValidationError(
            f"Unknown top-level setting(s) {unknown}. "
            f"Valid names: {sorted(top_fields - {'config_path'})}."
        )

    if "study_name" in raw:
        cfg.study_name = str(raw["study_name"])
    if "paths" in raw:
        _update_dataclass(cfg.paths, raw["paths"] or {})
    if "crs" in raw:
        _update_dataclass(cfg.crs, raw["crs"] or {})
    if "units" in raw:
        _update_dataclass(cfg.units, raw["units"] or {})
    if "extract" in raw:
        _update_dataclass(cfg.extract, raw["extract"] or {})
    if "lambda_calc" in raw:
        _update_dataclass(cfg.lambda_calc, raw["lambda_calc"] or {})
    if "labeler" in raw:
        _update_dataclass(cfg.labeler, raw["labeler"] or {})

    _coerce_types(cfg)
    cfg.units.vertical = normalize_vertical_units(str(cfg.units.vertical))

    base = config_path.parent
    cfg.paths.dem = _resolve_path(Path(cfg.paths.dem), base)
    cfg.paths.centerline = _resolve_path(Path(cfg.paths.centerline), base)
    cfg.paths.channel_polygon = _resolve_path(_as_path(cfg.paths.channel_polygon), base)
    cfg.paths.centerline_points = _resolve_path(_as_path(cfg.paths.centerline_points), base)
    cfg.paths.clip_boundary = _resolve_path(_as_path(cfg.paths.clip_boundary), base)
    cfg.paths.levee = _resolve_path(_as_path(cfg.paths.levee), base)
    cfg.paths.output_dir = _resolve_path(Path(cfg.paths.output_dir), base)
    if cfg.lambda_calc.labels_dir is not None:
        cfg.lambda_calc.labels_dir = _resolve_path(Path(cfg.lambda_calc.labels_dir), base)
    return cfg


def _coerce_types(cfg: WorkflowConfig) -> None:
    cfg.paths.dem = Path(cfg.paths.dem)
    cfg.paths.centerline = Path(cfg.paths.centerline)
    cfg.paths.channel_polygon = _as_path(cfg.paths.channel_polygon)
    cfg.paths.centerline_points = _as_path(cfg.paths.centerline_points)
    cfg.paths.clip_boundary = _as_path(cfg.paths.clip_boundary)
    cfg.paths.levee = _as_path(cfg.paths.levee)
    cfg.paths.output_dir = Path(cfg.paths.output_dir)
    if cfg.lambda_calc.labels_dir is not None:
        cfg.lambda_calc.labels_dir = Path(cfg.lambda_calc.labels_dir)

    ext = cfg.extract
    ext.spacing = float(ext.spacing)
    ext.cross_half_length = float(ext.cross_half_length)
    ext.max_half_length = float(ext.max_half_length)
    ext.sample_step = _optional_float(ext.sample_step)
    ext.valley_wall_dz = float(ext.valley_wall_dz)
    ext.valley_wall_persist = float(ext.valley_wall_persist)
    ext.valley_wall_skip_channel = float(ext.valley_wall_skip_channel)
    ext.valley_wall_keep = float(ext.valley_wall_keep)
    ext.max_elev_above_channel = _optional_float(ext.max_elev_above_channel)
    ext.channel_elev_window = float(ext.channel_elev_window)
    ext.backwater_length = float(ext.backwater_length)
    ext.centerline_smooth_window = float(ext.centerline_smooth_window)
    ext.centerline_smooth_sample = float(ext.centerline_smooth_sample)
    ext.flow_tangent_half_delta = float(ext.flow_tangent_half_delta)
    ext.slope_min = float(ext.slope_min)
    ext.default_slope = float(ext.default_slope)
    ext.slope_smooth_window = float(ext.slope_smooth_window)
    ext.slope_sample_step = float(ext.slope_sample_step)
    ext.thalweg_half_width = float(ext.thalweg_half_width)
    cfg.lambda_calc.auto_minmax = bool(cfg.lambda_calc.auto_minmax)
    cfg.labeler.skip_existing = bool(cfg.labeler.skip_existing)
    cfg.labeler.overlay_existing = bool(cfg.labeler.overlay_existing)
    cfg.labeler.levee_buffer = float(cfg.labeler.levee_buffer)


def apply_cli_overrides(cfg: WorkflowConfig, overrides: Mapping[str, Any]) -> WorkflowConfig:
    """Apply non-None CLI values on top of a loaded config."""
    mapping = {
        "dem": ("paths", "dem"),
        "centerline": ("paths", "centerline"),
        "channel_polygon": ("paths", "channel_polygon"),
        "centerline_points": ("paths", "centerline_points"),
        "clip_boundary": ("paths", "clip_boundary"),
        "levee": ("paths", "levee"),
        "output_dir": ("paths", "output_dir"),
        "study_name": ("study_name",),
        "crs_target": ("crs", "target"),
        "vertical_units": ("units", "vertical"),
        "spacing": ("extract", "spacing"),
        "auto_minmax": ("lambda_calc", "auto_minmax"),
        "labels_dir": ("lambda_calc", "labels_dir"),
    }
    for key, value in overrides.items():
        if value is None:
            continue
        path = mapping.get(key)
        if path is None:
            continue
        if path == ("study_name",):
            cfg.study_name = str(value)
            continue
        obj = cfg
        for part in path[:-1]:
            obj = getattr(obj, part)
        setattr(obj, path[-1], value)
    _coerce_types(cfg)
    cfg.units.vertical = normalize_vertical_units(str(cfg.units.vertical))
    return cfg


def config_to_dict(cfg: WorkflowConfig) -> dict:
    data = asdict(cfg)
    data.pop("config_path", None)
    return data
