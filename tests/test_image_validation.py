"""Unit tests for image validation module."""

import numpy as np
import pytest
from src.image_validation import ImageValidator

@pytest.fixture
def validator():
    return ImageValidator(min_dim=100, min_blur_score=50.0, min_brightness=30.0, max_brightness=230.0)

def test_valid_image(validator):
    # Sharp, well-lit synthetic checkerboard image with alternating 30x30 blocks
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    for y in range(0, 300, 30):
        for x in range(0, 300, 30):
            if ((x // 30) + (y // 30)) % 2 == 0:
                img[y:y+30, x:x+30] = 180
            else:
                img[y:y+30, x:x+30] = 90
    report = validator.validate_image(img)
    assert report.is_sharp is True
    assert report.is_valid is True

def test_too_small_image(validator):
    img = np.ones((50, 50, 3), dtype=np.uint8) * 128
    report = validator.validate_image(img)
    assert report.is_valid is False
    assert any("below minimum" in r for r in report.rejection_reasons)

def test_dark_image(validator):
    img = np.ones((200, 200, 3), dtype=np.uint8) * 10
    report = validator.validate_image(img)
    assert report.is_valid is False
    assert any("underexposed" in r for r in report.rejection_reasons)

def test_washed_out_image(validator):
    img = np.ones((200, 200, 3), dtype=np.uint8) * 250
    report = validator.validate_image(img)
    assert report.is_valid is False
    assert any("overexposed" in r for r in report.rejection_reasons)
