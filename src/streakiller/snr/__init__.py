"""
Per-streak SNR estimation via aperture photometry.

Public API::

    from streakiller.snr import StreakSNREstimator
    estimator = StreakSNREstimator()
    snr_list = estimator.estimate_all(image.data, final_lines, config.snr_params)
"""
from streakiller.snr.estimator import StreakSNREstimator

__all__ = ["StreakSNREstimator"]
