"""Unit tests for foot measurement and quality scoring."""

import numpy as np
import pytest
from src.measurement import measure_foot, MIN_PLAUSIBLE_FOOT_MM, MAX_PLAUSIBLE_FOOT_MM
from src.calibration import CalibrationData

@pytest.fixture
def mock_calibration():
    # 2.0 pixels per mm without homography
    return CalibrationData(
        pixels_per_mm=2.0,
        reference_object="Standard Card",
        reference_dims_mm=(85.6, 53.98),
        card_corners=None,
        homography_matrix=None,
        is_perspective_corrected=False,
        calibration_quality="High",
        quality_notes="Synthetic test calibration"
    )

def test_measurement_plausible(mock_calibration):
    # Heel at (100, 100), Toe at (100, 620) -> 520 pixels
    # At 2 px/mm -> 260.0 mm = 26.0 cm (Size 8 range)
    heel = (100.0, 100.0)
    toe = (100.0, 620.0)
    dummy_mask = np.zeros((800, 800), dtype=np.uint8)

    res = measure_foot(
        foot_mask=dummy_mask,
        heel_point=heel,
        toe_point=toe,
        calibration_data=mock_calibration,
        segmentation_confidence=0.92,
        pixel_width=180.0,
        image_shape=(800, 800)
    )

    assert pytest.approx(res.length_mm, 0.5) == 260.0
    assert pytest.approx(res.length_cm, 0.1) == 26.0
    assert pytest.approx(res.width_mm, 0.5) == 90.0
    assert res.is_physically_plausible is True
    assert res.reliability_status == "Reliable"

def test_measurement_too_short(mock_calibration):
    # 200 pixels at 2 px/mm = 100 mm = 10 cm (too small for adult foot)
    heel = (100.0, 100.0)
    toe = (100.0, 300.0)
    dummy_mask = np.zeros((800, 800), dtype=np.uint8)

    res = measure_foot(
        foot_mask=dummy_mask,
        heel_point=heel,
        toe_point=toe,
        calibration_data=mock_calibration
    )

    assert res.is_physically_plausible is False
    assert res.reliability_status == "Unreliable / Rejected"
    assert any("below physiological" in r for r in res.rejection_reasons)
