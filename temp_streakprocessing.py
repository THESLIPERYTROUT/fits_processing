import os
import cv2
import numpy as np
from astropy.io import fits
import json
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import logging
from PIL import Image 
from PIL import PngImagePlugin
from datetime import datetime
from satprocessing import build_satellite, get_ra_dec_rates
from image_calibrator import calibrate_image
from typing import Callable, List

Processor = Callable[['astro_image'], 'astro_image']

config_data= open('config.json')
config = json.load(config_data)

class astro_image:
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.metadata = {}
        self.image_data = None
        self.calibrated_image_data = None
        self.streaks = []
        self.satellites = []
        self.ra_dec_rates = None

    def load_image(self):
        try:
            with fits.open(self.filepath) as hdul:
                self.image_data = hdul[0].data.astype(np.float32)
                header = hdul[0].header
                self.metadata['exposure_time'] = header.get('EXPTIME', None)
                self.metadata['observation_time'] = header.get('DATE-OBS', None)
                self.metadata['telescope'] = header.get('TELESCOP', None)
                self.metadata['instrument'] = header.get('INSTRUME', None)
                logging.info(f"Loaded image {self.filename} with shape {self.image_data.shape}")
        except Exception as e:
            logging.error(f"Error loading image {self.filename}: {e}")
            raise e 
        
    def calibrate_image(self, bias_frame=None, dark_frame=None, flat_frame=None):
        try:
            self.calibrated_image_data = calibrate_image(self.image_data, bias_frame, dark_frame, flat_frame)
            logging.info(f"Calibrated image {self.filename}")
        except Exception as e:
            logging.error(f"Error calibrating image {self.filename}: {e}")
            raise e
        
    def display_image(self):
        if self.calibrated_image_data is not None:
            plt.imshow(self.calibrated_image_data, cmap='gray', origin='lower')
            plt.title(f"Calibrated Image: {self.filename}")
            plt.colorbar()
            plt.show()

    def save_image_as(self, output_path, output_format='FITS'):
        try:
            if output_format == 'PNG':
                img = Image.fromarray(self.image_data)
                img.save(output_path)
                logging.info(f"Saved calibrated image as {output_path}")
            elif output_format == 'FITS':
                hdu = fits.PrimaryHDU(self.calibrated_image_data)
                hdu.writeto(output_path, overwrite=True)
            logging.info(f"Saved calibrated image as {output_path}")
        except Exception as e:
            logging.error(f"Error saving image {self.filename} as {output_path}: {e}")
            raise e

class pipeline:
    def __init__(self, processors: List[Processor]):
        self.processors = processors

    def add(self, p: Processor):
        self.steps.append(p)

    def process(self, astro_img: astro_image) -> astro_image:
        for processor in self.processors:
            try:
                astro_img = processor(astro_img)
            except Exception as e:
                logging.error(f"Error processing image {astro_img.filename} with processor {processor.__name__}: {e}")
                raise e
        return astro_img



streak_pipeline = pipeline([])

pipeline.add_processor(astro_image.load_image)
pipeline.add_processor(check_config(filepath))

# CORE PROCESS FUNCTIONS
def detect_streaks(image, minLinelength, enabled_filters, background_dectection_method) -> list:

    p_low, p_high = np.percentile(image, (2, 98))
    data_clipped = np.clip(image, p_low, p_high)

    # Normalize to 0–255
    norm_data = (data_clipped - p_low) / (p_high - p_low) * 255
    save_image_with_metadata("normalized_image.png", norm_data, 
                             tags={"stage": "normalized", "min": str(np.min(image)), "max": str(np.max(image))})
    
    image_display = np.uint8(norm_data)
    print(f"Image dtype: {image.dtype}, min: {np.min(image)}, max: {np.max(image)}")
    save_image_with_metadata("image_display.png", image_display, 
                             tags={"stage": "normalized", "min": str(np.min(image)), "max": str(np.max(image))})


    if background_dectection_method.get("simple_median", True):
        binary = simple_median(image)
        background_dectection_method = "simple_median"
    elif background_dectection_method.get("Guassian_blur", True):  
        binary = gaussian_binarize(image)
        background_dectection_method = "Guassian_blur"
    elif background_dectection_method.get("doublepass_median_to_guassian_blur", True):
        binary = double_threshold(image)
        background_dectection_method = "doublepass_median_to_guassian_blur"
    else:
        logging.error("No valid background detection method selected, defaulting to Gaussian blur")
        binary = gaussian_binarize(image)
        background_dectection_method = "Guassian_blur"

    # Use Hough Line Transform to detect streaks
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=60,
                            minLineLength=minLinelength, maxLineGap=5)
    if lines is None:
        return logging.error("No lines detected"), image_display
    print(f"Detected {len(lines)} lines") 

    enabled_filters_tags = []

    # collect snapshots for visualization: list of (stage_name, lines)
    stages = []
    stages.append(("detected", lines))
    current_lines = lines

    if enabled_filters.get("midpoint_filter", True):
        new_lines = midpoint_filter_close_lines(current_lines, min_distance=10)
        logging.info(f"{len(new_lines)} lines after midpoint filtering")
        enabled_filters_tags.append("midpoint_filter")
        stages.append(("midpoint_filter", new_lines))
        current_lines = new_lines

    if enabled_filters.get("line_angle", True):
        new_lines = line_angle_filter(current_lines, min_angle_diff=10)
        logging.info(f"{len(new_lines)} lines after filtering by angle")
        enabled_filters_tags.append("line_angle")
        stages.append(("line_angle", new_lines))
        current_lines = new_lines

    if enabled_filters.get("colinear_filter", True):
        new_lines = add_colinear_segments(current_lines)
        print(f"{len(new_lines)} lines after merging collinear segments")
        enabled_filters_tags.append("colinear_filter")
        stages.append(("colinear_filter", new_lines))
        current_lines = new_lines

    if enabled_filters.get("endpoint_filer", True):
        new_lines = endpoint_filer(current_lines, min_distance=10)
        logging.info(f"{len(new_lines)} lines after endpoint filtering")
        enabled_filters_tags.append("endpoint_filer")
        stages.append(("endpoint_filer", new_lines))
        current_lines = new_lines

    if enabled_filters.get("length_filter", True):
        new_lines = line_length_filter(current_lines)
        logging.info(f"{len(new_lines)} lines after length filtering")
        enabled_filters_tags.append("length_filter")
        stages.append(("length_filter", new_lines))
        current_lines = new_lines

    # final filtered lines
    filtered_lines = current_lines

    # create a visualization overlay showing evolution across stages
    try:
        annotated = draw_filter_stage_overlays(binary, stages)
        save_image_with_metadata("filter_stage_overlays.png", annotated, tags={"stage": "filter_evolution"}, dir="output")
    except Exception as e:
        logging.warning(f"Could not create filter-stage overlay: {e}")

    tags = ["background_method=" + background_dectection_method,
            "enabled_filters=" + ",".join(enabled_filters_tags), "processed at" + str(datetime.now())]
    labels = [f"{i+1}" for i in range(len(filtered_lines))]

    output = cv2.cvtColor(image_display, cv2.COLOR_GRAY2BGR)  # make color image to draw on
    output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    filtered_lines = np.array(filtered_lines)
    if filtered_lines is not None:
        for line in filtered_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
    

    return filtered_lines, image_display, labels, tags
