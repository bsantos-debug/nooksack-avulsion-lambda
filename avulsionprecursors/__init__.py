"""Avulsion precursor analysis from DEM cross-sections.

Public workflow
---------------
Load a YAML config, then:

    avulsion-process extract --config config.yaml
    avulsion-process label   --config config.yaml
    avulsion-process lambda  --config config.yaml
"""

from avulsionprecursors.config import WorkflowConfig, load_config
from avulsionprecursors.pipeline.extract import extract_cross_sections
from avulsionprecursors.pipeline.lambda_calc import calculate_lambda

__all__ = [
    "WorkflowConfig",
    "load_config",
    "extract_cross_sections",
    "calculate_lambda",
]
