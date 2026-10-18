"""End-to-end integration tests for YOLO model loading and foot segmentation."""

from pathlib import Path
import pytest
import cv2
import numpy as np

from src.model_loader import ModelLoader
from src.foot_segmentation import FootSegmenter

def test_model_loader_caching():
    model1 = ModelLoader.load_model("models/best.pt")
    model2 = ModelLoader.load_model("models/best.pt")
    assert model1 is model2
    info = ModelLoader.get_model_info()
    assert info["loaded"] is True
    assert info["task"] == "segment"
    assert "foot" in info["classes"].values()

def test_foot_segmenter_on_real_image():
    test_imgs = list(Path("test_images").glob("*.jpg"))
    if not test_imgs:
        pytest.skip("No test images in test_images/ directory.")

    sample_img_path = str(test_imgs[0])
    segmenter = FootSegmenter(model_path="models/best.pt")
    primary_foot = segmenter.get_primary_foot(sample_img_path)

    assert primary_foot is not None
    assert primary_foot.confidence > 0.40
    assert primary_foot.mask is not None
    assert primary_foot.area_pixels > 500

    # Test overlay generation
    img_bgr = cv2.imread(sample_img_path)
    overlay = segmenter.draw_overlay(img_bgr, primary_foot)
    assert overlay.shape == img_bgr.shape
