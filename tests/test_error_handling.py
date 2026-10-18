"""Tests for error handling, missing files, corrupt inputs, and edge cases."""

import pytest
import numpy as np
from src.model_loader import ModelLoader, ModelNotFoundError
from src.calibration import ReferenceCardCalibrator
from src.foot_landmarks import FootLandmarkExtractor
from src.image_validation import ImageValidator

def test_missing_model_file():
    with pytest.raises(ModelNotFoundError):
        ModelLoader.load_model("non_existent_dir/definitely_not_a_model.pt", force_reload=True)

def test_missing_reference_card():
    calibrator = ReferenceCardCalibrator()
    # Plain image with no rectangles
    blank_img = np.ones((500, 500, 3), dtype=np.uint8) * 128
    calib = calibrator.calibrate(blank_img)
    assert calib is None

def test_partially_visible_truncated_foot():
    extractor = FootLandmarkExtractor()
    # Mask touching the top and bottom borders (h=500, w=500)
    truncated_mask = np.zeros((500, 500), dtype=np.uint8)
    truncated_mask[0:499, 200:300] = 255  # Touches y=0 and y=499
    landmarks = extractor.extract_landmarks(truncated_mask)
    if landmarks is not None:
        # Should be flagged as invalid or touching boundary
        assert landmarks.is_valid is False or "boundary" in landmarks.quality_notes

def test_blurry_image_rejection():
    validator = ImageValidator(min_blur_score=50.0)
    # Extremely blurry gradient
    blurry = np.zeros((400, 400, 3), dtype=np.uint8)
    for i in range(400):
        blurry[i, :] = int((i / 400.0) * 200)
    report = validator.validate_image(blurry)
    assert report.is_sharp is False
    assert report.blur_score < 50.0
