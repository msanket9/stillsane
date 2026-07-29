"""The comparison engine.

Deliberately isolated from I/O: everything in here takes `Sample` objects and
returns verdicts. No HTTP, no filesystem, no config parsing. That boundary is
what makes the hard part testable against fixtures with no network and no spend.
"""

from .pooling import PoolConfig, is_clean, merge_pool, original_scale_of, pool_from_run
from .variance import (
    BandConfig,
    evaluate,
    evaluate_categorical,
    evaluate_pairwise,
    evaluate_pointwise,
    mann_whitney_p,
    pairwise_cross,
    pairwise_within,
    robust_band,
    robust_centre_scale,
    z_score,
)
from .verdict import build_run, compare_probe

__all__ = [
    "BandConfig",
    "PoolConfig",
    "build_run",
    "compare_probe",
    "evaluate",
    "evaluate_categorical",
    "evaluate_pairwise",
    "evaluate_pointwise",
    "is_clean",
    "mann_whitney_p",
    "merge_pool",
    "original_scale_of",
    "pairwise_cross",
    "pairwise_within",
    "pool_from_run",
    "robust_band",
    "robust_centre_scale",
    "z_score",
]
