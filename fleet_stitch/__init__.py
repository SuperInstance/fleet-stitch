"""fleet-stitch — Constraint manifold projection for model activation spaces."""

from .manifold import ManifoldProjector
from .registry import StitchRegistry

__all__ = ["ManifoldProjector", "StitchRegistry"]
__version__ = "0.1.0"
