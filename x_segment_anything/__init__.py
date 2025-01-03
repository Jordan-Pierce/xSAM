"""Top-level package for xSAM."""

__author__ = """Jordan Pierce"""
__email__ = "jordan.pierce@noaa.gov"
__version__ = "0.0.7"


from .build_sam import (
    build_sam,
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam_vit_b,
    build_sam_vit_t,
    sam_model_registry,
    sam_model_urls
)
from .predictor import SamPredictor
from .automatic_mask_generator import SamAutomaticMaskGenerator