"""TruthfulQA modules used by the API-free ITI environment.

The upstream package eagerly imports its legacy metrics module, which depends
on ``datasets.load_metric`` removed from recent datasets releases. Metrics are
loaded lazily by the ITI evaluation path only when a legacy metric is requested.
"""

from . import configs, models, presets, utilities

__all__ = ["configs", "models", "presets", "utilities"]
