"""Image validation and quality control module.

Ensures photographic inputs meet resolution, sharpness, exposure, and boundary
integrity criteria before triggering AI segmentation and measurement.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

@dataclass
class ImageValidationReport:
    """Comprehensive validation diagnostic report."""
    is_valid: bool
    width: int
    height: int
    blur_score: float                # Laplacian variance (higher = sharper)
    mean_brightness: float           # 0 - 255 grayscale average
    is_sharp: bool
    is_well_lit: bool
    warnings: List[str] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)


class ImageValidator:
    """Validates photographic conditions for accurate foot photogrammetry."""

    def __init__(
        self,
        min_dim: int = 300,
        max_dim: int = 4096,
        min_blur_score: float = 50.0,
        min_brightness: float = 35.0,
        max_brightness: float = 235.0
    ):
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.min_blur_score = min_blur_score
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    def validate_image(self, image_bgr: np.ndarray) -> ImageValidationReport:
        """Runs pre-inference quality checks on the input image."""
        if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
            return ImageValidationReport(
                is_valid=False,
                width=0,
                height=0,
                blur_score=0.0,
                mean_brightness=0.0,
                is_sharp=False,
                is_well_lit=False,
                rejection_reasons=["Image data is empty, corrupt, or could not be decoded."]
            )

        h, w = image_bgr.shape[:2]
        warnings = []
        rejections = []

        # 1. Dimensions check
        if w < self.min_dim or h < self.min_dim:
            rejections.append(
                f"Image resolution ({w}x{h}) is below minimum requirement ({self.min_dim}x{self.min_dim})."
            )
        if w > self.max_dim or h > self.max_dim:
            warnings.append(
                f"Image resolution ({w}x{h}) is very large; downsizing may improve speed."
            )

        # 2. Convert to grayscale for sharpness and brightness analysis
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # 3. Blur detection (Laplacian variance)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_sharp = blur_score >= self.min_blur_score

        if not is_sharp:
            if blur_score < 20.0:
                rejections.append(
                    f"Image is heavily blurred (sharpness index: {blur_score:.1f}, min: {self.min_blur_score}). "
                    f"Please keep the camera steady and refocus."
                )
            else:
                warnings.append(
                    f"Image has moderate motion blur (sharpness: {blur_score:.1f}). Landmark precision may be reduced."
                )

        # 4. Brightness / exposure assessment
        mean_brightness = float(np.mean(gray))
        is_well_lit = (self.min_brightness <= mean_brightness <= self.max_brightness)

        if mean_brightness < self.min_brightness:
            rejections.append(
                f"Photograph is severely underexposed/dark (brightness: {mean_brightness:.1f}/255). "
                f"Please ensure good ambient lighting."
            )
        elif mean_brightness > self.max_brightness:
            rejections.append(
                f"Photograph is severely overexposed/washed out (brightness: {mean_brightness:.1f}/255). "
                f"Avoid direct blinding camera flash."
            )
        elif mean_brightness < 60.0:
            warnings.append("Lighting is somewhat dim; high contrast lighting is recommended.")

        is_valid = len(rejections) == 0

        return ImageValidationReport(
            is_valid=is_valid,
            width=w,
            height=h,
            blur_score=blur_score,
            mean_brightness=mean_brightness,
            is_sharp=is_sharp,
            is_well_lit=is_well_lit,
            warnings=warnings,
            rejection_reasons=rejections
        )
