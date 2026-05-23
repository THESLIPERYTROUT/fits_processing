from streakiller.detection.detector import RawDetection, StreakDetector
from streakiller.detection.fft_detector import FftCorrelationDetector
from streakiller.detection.peak_hough_detector import PeakHoughDetector

__all__ = [
    "RawDetection",
    "StreakDetector",
    "FftCorrelationDetector",
    "PeakHoughDetector",
]
