"""Unit tests for reference object calibration module."""

import numpy as np
import cv2
import pytest
from src.calibration import ReferenceCardCalibrator, STANDARD_CARD_WIDTH_MM, STANDARD_CARD_HEIGHT_MM

@pytest.fixture
def calibrator():
    return ReferenceCardCalibrator()

def test_corner_ordering(calibrator):
    # Unordered corners of a 100x60 rectangle
    pts = np.array([
        [150.0, 110.0], # BR
        [50.0, 50.0],   # TL
        [50.0, 110.0],  # BL
        [150.0, 50.0]   # TR
    ])
    ordered = calibrator.order_corners(pts)
    assert np.allclose(ordered[0], [50.0, 50.0])   # TL
    assert np.allclose(ordered[1], [150.0, 50.0])  # TR
    assert np.allclose(ordered[2], [150.0, 110.0]) # BR
    assert np.allclose(ordered[3], [50.0, 110.0])  # BL

def test_calibrate_from_corners(calibrator):
    # Perfect horizontal card of 85.6mm x 53.98mm at 2 pixels per mm
    ppm = 2.0
    w_px = STANDARD_CARD_WIDTH_MM * ppm
    h_px = STANDARD_CARD_HEIGHT_MM * ppm

    corners = np.array([
        [100.0, 100.0],
        [100.0 + w_px, 100.0],
        [100.0 + w_px, 100.0 + h_px],
        [100.0, 100.0 + h_px]
    ])

    calib = calibrator.calibrate_from_corners(corners, (800, 800))
    assert calib.is_perspective_corrected is True
    assert pytest.approx(calib.pixels_per_mm, rel=1e-2) == 2.0
    assert calib.calibration_quality in ["High", "Moderate"]

def test_homography_point_transformation(calibrator):
    ppm = 3.0
    w_px = STANDARD_CARD_WIDTH_MM * ppm
    h_px = STANDARD_CARD_HEIGHT_MM * ppm

    corners = np.array([
        [0.0, 0.0],
        [w_px, 0.0],
        [w_px, h_px],
        [0.0, h_px]
    ])

    calib = calibrator.calibrate_from_corners(corners, (600, 600))
    test_pts = np.array([
        [0.0, 0.0],
        [w_px, 0.0]
    ])
    trans = calibrator.transform_points(test_pts, calib.homography_matrix)
    # Distance between them in mm should equal STANDARD_CARD_WIDTH_MM
    dist_mm = np.linalg.norm(trans[1] - trans[0])
    assert pytest.approx(dist_mm, rel=1e-2) == STANDARD_CARD_WIDTH_MM

def test_auto_detect_synthetic_card(calibrator):
    # Create black canvas and draw white card with standard ratio 85.6 : 53.98
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    card_w, card_h = 171, 108  # Ratio 1.583
    cv2.rectangle(img, (200, 200), (200 + card_w, 200 + card_h), (255, 255, 255), -1)

    detected = calibrator.auto_detect_card(img)
    assert detected is not None
    assert len(detected) == 4
