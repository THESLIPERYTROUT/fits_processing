# All magic numbers extracted from the original codebase, documented with their source.
# These are the default values; all are overridable via PipelineConfig.

# --- Hough Line Transform (streakprocessing.py:112) ---
HOUGH_THRESHOLD = 60          # minimum vote count for a line to be accepted
HOUGH_MAX_LINE_GAP = 5        # max pixel gap allowed within a single line
HOUGH_RHO = 1.0               # distance resolution (pixels)
HOUGH_THETA_DEG = 1.0         # angle resolution (degrees)

# --- Background: Gaussian blur (streakprocessing.py:315, 330, 347-358) ---
GAUSSIAN_KERNEL_SIZE = 51     # blur kernel size for background estimation
GAUSSIAN_SIGMA_LADDER = (3.0, 2.5, 2.0, 1.5, 1.2)  # k-values tried in order
GAUSSIAN_MIN_BINARY_PIXELS = 50  # min foreground pixels before accepting a binary
MAD_NORMALIZATION_FACTOR = 1.4826  # converts MAD to consistent sigma estimate

# --- Background: Simple median (streakprocessing.py:394) ---
SIMPLE_MEDIAN_SIGMA_MULT = 1.2    # threshold = median + mult * stddev
SIMPLE_MEDIAN_MORPH_KERNEL = 5    # morphological close kernel size

# --- Background: Double-pass threshold (streakprocessing.py:448, 469, 471) ---
DOUBLE_PASS_SIGMA_MULT = 2.0      # threshold = median + mult * stddev (both passes)
DOUBLE_PASS_MORPH_KERNEL = 5      # morphological close kernel size
DOUBLE_PASS_INPAINT_RADIUS = 3    # Telea inpainting neighbourhood radius (pixels)

# --- Filters (streakprocessing.py:126, 133, 147, 154) ---
MIDPOINT_MIN_DISTANCE = 10.0      # pixels; remove lines whose midpoints are closer than this
ENDPOINT_MIN_DISTANCE = 10.0      # pixels; remove lines whose endpoints are closer than this
ANGLE_MIN_DIFF_DEG = 10.0         # degrees; deduplicate lines within this angle of each other
LENGTH_FRACTION = 0.9            # keep lines >= this fraction of the longest detected line
COLINEAR_ORIENTATION_TOL = 1.0    # cross-product magnitude below which two segments are collinear

# --- Hot pixel removal (streakprocessing.py:685) ---
HOTPIXEL_THRESHOLD = 5000         # ADU; pixels above this are considered hot

# --- Image normalisation for display (streakprocessing.py:83-88) ---
NORM_PERCENTILE_LOW = 2.0         # lower percentile clip
NORM_PERCENTILE_HIGH = 98.0       # upper percentile clip
