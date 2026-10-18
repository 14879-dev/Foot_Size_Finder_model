"""Foot measurement and quality control module.

Combines foot segmentation masks, anatomical endpoints, and calibration data
to compute calibrated real-world measurements with comprehensive reliability scoring.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from src.calibration import CalibrationData, ReferenceCardCalibrator

# Biological limits for adult human foot lengths (in mm)
MIN_PLAUSIBLE_FOOT_MM = 180.0  # 18.0 cm (roughly US Kids 12 / EU 29)
MAX_PLAUSIBLE_FOOT_MM = 330.0  # 33.0 cm (roughly US Men 15 / EU 49)

@dataclass
class FootMeasurementResult:
    """Standardized measurement outcome with reliability indices and warnings."""
    length_mm: float
    length_cm: float
    width_mm: float
    width_cm: float
    heel_point: Tuple[float, float]
    toe_point: Tuple[float, float]
    pixel_length: float
    pixel_width: float
    pixels_per_mm: float
    calibration_method: str
    segmentation_confidence: float
    reliability_status: str       # 'Reliable', 'Caution / Approximate', 'Unreliable / Rejected'
    is_physically_plausible: bool
    warnings: List[str] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)
    summary_message: str = ""


def measure_foot(
    foot_mask: np.ndarray,
    heel_point: Tuple[float, float],
    toe_point: Tuple[float, float],
    calibration_data: CalibrationData,
    segmentation_confidence: float = 1.0,
    pixel_width: Optional[float] = None,
    image_shape: Optional[Tuple[int, int]] = None
) -> FootMeasurementResult:
    """Measures physical foot dimensions using calibrated photogrammetry.

    Args:
        foot_mask: Binary mask of foot.
        heel_point: (x, y) coordinates of heel landmark in pixel space.
        toe_point: (x, y) coordinates of toe landmark in pixel space.
        calibration_data: CalibrationData object from reference card.
        segmentation_confidence: AI model confidence score.
        pixel_width: Foot width in pixels perpendicular to axis.
        image_shape: (height, width) of input image.

    Returns:
        FootMeasurementResult with metric lengths, quality status, and advisory notes.
    """
    warnings: List[str] = []
    rejections: List[str] = []

    # 1. Pixel Euclidean distance
    dx = toe_point[0] - heel_point[0]
    dy = toe_point[1] - heel_point[1]
    pixel_length = float(np.sqrt(dx * dx + dy * dy))

    if pixel_length < 20.0:
        rejections.append("Measured pixel distance between heel and toe is unrealistically short.")

    # 2. Convert to millimeters via Homography or PPM
    if calibration_data.is_perspective_corrected and calibration_data.homography_matrix is not None:
        try:
            pts_img = np.array([heel_point, toe_point], dtype=np.float32)
            pts_mm = ReferenceCardCalibrator.transform_points(pts_img, calibration_data.homography_matrix)
            dx_mm = pts_mm[1][0] - pts_mm[0][0]
            dy_mm = pts_mm[1][1] - pts_mm[0][1]
            length_mm = float(np.sqrt(dx_mm * dx_mm + dy_mm * dy_mm))
            calib_method = "Perspective Homography (Planar 2D Metric Space)"
        except Exception as e:
            # Fallback to PPM
            length_mm = pixel_length / calibration_data.pixels_per_mm
            calib_method = f"Direct Linear Scaling ({calibration_data.pixels_per_mm:.2f} px/mm) [Homography Fallback: {e}]"
            warnings.append("Homography projection failed; reverted to linear scaling.")
    else:
        ppm = max(0.001, calibration_data.pixels_per_mm)
        length_mm = pixel_length / ppm
        calib_method = f"Direct Linear Scaling ({ppm:.2f} px/mm)"

    length_cm = round(length_mm / 10.0, 2)

    # 3. Foot width in mm
    if pixel_width is not None and pixel_width > 0:
        width_mm = pixel_width / max(0.001, calibration_data.pixels_per_mm)
        width_cm = round(width_mm / 10.0, 2)
    else:
        width_mm = 0.0
        width_cm = 0.0

    # 4. Physical Plausibility Validation
    is_plausible = (MIN_PLAUSIBLE_FOOT_MM <= length_mm <= MAX_PLAUSIBLE_FOOT_MM)
    if not is_plausible:
        if length_mm < MIN_PLAUSIBLE_FOOT_MM:
            rejections.append(
                f"Calculated foot length ({length_cm} cm) is below physiological adult bounds (< {MIN_PLAUSIBLE_FOOT_MM/10:.1f} cm). "
                f"Check card placement and ensure card and foot are resting on the exact same flat plane."
            )
        else:
            rejections.append(
                f"Calculated foot length ({length_cm} cm) exceeds normal physiological bounds (> {MAX_PLAUSIBLE_FOOT_MM/10:.1f} cm). "
                f"Check card placement and ensure the camera is directly above."
            )

    # 5. Boundary Proximity Check
    if image_shape is not None:
        img_h, img_w = image_shape
        margin = 10
        for pt_name, pt in [("Heel", heel_point), ("Toe", toe_point)]:
            if pt[0] < margin or pt[0] > img_w - margin or pt[1] < margin or pt[1] > img_h - margin:
                warnings.append(f"{pt_name} is near image perimeter; partial truncation may compromise accuracy.")

    # 6. Quality Status Determination
    if rejections:
        reliability = "Unreliable / Rejected"
        summary = "Measurement could not be verified due to physical or geometric inconsistencies."
    elif calibration_data.calibration_quality == "Low" or warnings:
        reliability = "Caution / Approximate"
        summary = "Estimated foot measurement computed with moderate confidence. Verify camera was held parallel to floor."
    else:
        reliability = "Reliable"
        summary = "High-confidence photogrammetric measurement validated with standard reference calibration."

    return FootMeasurementResult(
        length_mm=round(length_mm, 1),
        length_cm=length_cm,
        width_mm=round(width_mm, 1),
        width_cm=width_cm,
        heel_point=heel_point,
        toe_point=toe_point,
        pixel_length=round(pixel_length, 1),
        pixel_width=round(pixel_width if pixel_width else 0.0, 1),
        pixels_per_mm=round(calibration_data.pixels_per_mm, 3),
        calibration_method=calib_method,
        segmentation_confidence=round(segmentation_confidence, 4),
        reliability_status=reliability,
        is_physically_plausible=is_plausible,
        warnings=warnings,
        rejection_reasons=rejections,
        summary_message=summary
    )
