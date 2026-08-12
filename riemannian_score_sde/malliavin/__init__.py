"""Malliavin teacher utilities shared by training and diagnostics."""

from .rao_blackwell import (
    rao_blackwell_estimate_s2,
    rao_blackwell_estimate_s2_batch,
    s2_parallel_transport,
)

__all__ = (
    "rao_blackwell_estimate_s2",
    "rao_blackwell_estimate_s2_batch",
    "s2_parallel_transport",
)
