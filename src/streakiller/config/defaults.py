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
LENGTH_FRACTION = 0.90            # keep lines +- this fraction of the mode of all detected line lengths
COLINEAR_ORIENTATION_TOL = 1.0    # cross-product magnitude below which two segments are collinear

# --- Hot pixel removal (streakprocessing.py:685) ---
HOTPIXEL_THRESHOLD = 5000         # ADU; pixels above this are considered hot

# --- Image normalisation for display (streakprocessing.py:83-88) ---
NORM_PERCENTILE_LOW = 2.0         # lower percentile clip
NORM_PERCENTILE_HIGH = 98.0       # upper percentile clip

# --- Background: Adaptive Local (local mesh + iterative sigma-clipping + local SNR) ---
ADAPTIVE_LOCAL_TILE_SIZE = 64         # side length of each mesh tile in pixels (must be >= 8)
ADAPTIVE_LOCAL_CLIP_SIGMA = 3.0       # per-tile sigma-clipping rejection threshold
ADAPTIVE_LOCAL_N_ITERATIONS = 3       # number of sigma-clipping passes per tile
ADAPTIVE_LOCAL_SNR_THRESHOLD = 2.0    # min local SNR (residual / local_sigma) for foreground
ADAPTIVE_LOCAL_MIN_TILE_PIXELS = 10   # min surviving pixels for a tile to be considered valid
ADAPTIVE_LOCAL_MORPH_KERNEL = 3       # morphological close kernel size (pixels)
ADAPTIVE_LOCAL_GAUSSIAN_KERNEL_SIZE = 51  # Gaussian high-pass pre-filter kernel (must be odd)

# --- Per-streak SNR estimation (aperture photometry on raw image) ---
SNR_HALF_WIDTH_PX = 3      # on-streak aperture half-width: samples ±N pixels from the centerline
SNR_OFF_GAP_PX = 3         # gap between on-streak edge and background band (pixels)
SNR_OFF_WIDTH_PX = 10      # width of each background band on either side of the streak (pixels)
SNR_MIN_OFF_PIXELS = 20    # minimum background pixels required; fewer yields NaN SNR

# --- FFT Correlation Detector (Streak_Detector.py) ---
FFT_THRESHOLD_SIGMA = 0.75       # correlation peak threshold: median + N * std
FFT_MIN_DISTANCE = 10            # minimum pixel distance between accepted peaks
FFT_MIN_TEMPLATE_AREA = 40       # minimum pixels for a feature to be a template candidate
FFT_TEMPLATE_PADDING = 10        # pixels of context padding around the template cutout
FFT_MAX_WIDTH_STD = 4.0          # max PCA width std; rejects blobs wider than N px sigma
FFT_MIN_ELONGATION = 3.0         # minimum length/width ratio; rejects near-circular objects
FFT_PERCENTILE_THRESHOLD = 99.0  # image percentile used to build the initial binary mask
FFT_TEMPLATE_EDGE_MARGIN = 15    # template candidates closer than N px to the image edge are rejected
FFT_STREAK_EDGE_MARGIN = 5       # detected streaks closer than N px to the image edge are rejected
FFT_PROMINENCE_FRACTION = 0.5    # reject peaks below this fraction of the maximum correlation score

# --- Peak-Hough detector ---
PEAK_HOUGH_THRESHOLD_SIGMA = 3.0        # binary mask: keep pixels above N·σ_global(residual)
PEAK_HOUGH_GAUSSIAN_KERNEL_SIZE = 51    # high-pass background kernel (must be odd)
PEAK_HOUGH_ENDPOINT_WALK_SIGMA = 1.5    # walk-out floor = N × σ_local at endpoint
PEAK_HOUGH_ENDPOINT_GAP_TOLERANCE = 3   # consecutive below-floor steps before stopping
