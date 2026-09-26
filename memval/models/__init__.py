"""
Abstract models and baselines for testing sequential memory architectures.
"""
from .base import HippocampalModel
from .capabilities import (
    OnlineTrainable,
    UnsupportedRegime,
    is_online_equivalent,
    supports_online,
)
from . import baselines

__all__ = [
    "HippocampalModel",
    "OnlineTrainable",
    "UnsupportedRegime",
    "supports_online",
    "is_online_equivalent",
    "baselines",
]
