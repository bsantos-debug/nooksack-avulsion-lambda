"""Beginner-friendly errors for missing files, units, CRS, and invalid inputs."""

from __future__ import annotations


class WorkflowError(Exception):
    """User-facing error with a plain-language message (no traceback required)."""


class InputValidationError(WorkflowError):
    """A path, parameter, CRS, or raster input is missing or incompatible."""


class UnitsError(InputValidationError):
    """Horizontal or vertical units are missing, conflicting, or unsupported."""


class CRSError(InputValidationError):
    """A dataset is missing a CRS, or the working CRS is not projected."""
