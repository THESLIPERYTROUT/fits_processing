"""
BackgroundEstimator protocol — all background estimators satisfy this interface.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from streakiller.config.schema import BackgroundParams


@runtime_checkable
class BackgroundEstimator(Protocol):
    """
    Converts a float32 2-D image into a uint8 binary foreground mask.

    Implementations must:
    - Return a uint8 array of the same spatial shape with values in {0, 255}
    - Not modify the input array
    - Not perform any I/O (no file writes)
    """

    def estimate(self, data: np.ndarray, params: BackgroundParams) -> np.ndarray:
        ...
