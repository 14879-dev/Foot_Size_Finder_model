"""Unit tests for foot landmark extraction."""

import numpy as np
import cv2
import pytest
from src.foot_landmarks import FootLandmarkExtractor

@pytest.fixture
def extractor():
    return FootLandmarkExtractor()

def create_synthetic_foot_mask(h=600, w=600, length_px=250, heel_w=50, toe_w=85):
    """Draws an anatomical foot-shaped silhouette with a narrower heel and wider forefoot."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2

    # Draw heel (narrow ellipse at bottom)
    cv2.ellipse(mask, (cx, cy + length_px // 3), (heel_w // 2, 40), 0, 0, 360, 255, -1)
    # Draw midfoot
    pts_mid = np.array([
        [cx - heel_w // 2, cy + length_px // 3],
        [cx + heel_w // 2, cy + length_px // 3],
        [cx + toe_w // 2, cy - length_px // 3],
        [cx - toe_w // 2, cy - length_px // 3]
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts_mid], 255)
    # Draw forefoot and toes (wider ellipse at top)
    cv2.ellipse(mask, (cx, cy - length_px // 3), (toe_w // 2, 50), 0, 0, 360, 255, -1)
    return mask

def test_landmark_extraction_synthetic_foot(extractor):
    mask = create_synthetic_foot_mask()
    landmarks = extractor.extract_landmarks(mask)

    assert landmarks is not None
    assert landmarks.length_pixels > 150.0
    assert landmarks.width_pixels > 30.0
    assert landmarks.aspect_ratio >= 1.5

    # Forefoot is at top (lower y), heel is at bottom (higher y)
    assert landmarks.toe_point[1] < landmarks.heel_point[1]
    assert landmarks.is_valid is True

def test_empty_or_small_mask(extractor):
    empty_mask = np.zeros((400, 400), dtype=np.uint8)
    res = extractor.extract_landmarks(empty_mask)
    assert res is None

    small_mask = np.zeros((400, 400), dtype=np.uint8)
    small_mask[100:105, 100:105] = 255
    res_small = extractor.extract_landmarks(small_mask)
    assert res_small is None
