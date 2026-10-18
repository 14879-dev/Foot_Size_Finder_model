"""Evaluation procedure for Khybersole AI Foot Size Finder.

Runs the segmentation and landmark pipeline over test images,
measuring inference speed, detection confidence, landmark plausibility,
and reporting comprehensive dataset statistics.
"""

from pathlib import Path
import time
import cv2
import numpy as np

from src.model_loader import ModelLoader
from src.foot_segmentation import FootSegmenter
from src.foot_landmarks import FootLandmarkExtractor
from src.calibration import ReferenceCardCalibrator
from src.size_recommender import SizeRecommender
from src.image_validation import ImageValidator

def evaluate_dataset(test_images_dir: str = "test_images", model_path: str = "models/best.pt"):
    print("=" * 65)
    print("KHYBERSOLE AI FOOT SIZE FINDER — BENCHMARK EVALUATION")
    print("=" * 65)

    img_paths = list(Path(test_images_dir).glob("*.jpg")) + list(Path(test_images_dir).glob("*.png"))
    total_samples = len(img_paths)
    print(f"[INFO] Evaluating {total_samples} test images from '{test_images_dir}'...")

    if total_samples == 0:
        print("[ERROR] No test images found.")
        return

    # Load components
    segmenter = FootSegmenter(model_path=model_path, conf_threshold=0.25)
    extractor = FootLandmarkExtractor()
    validator = ImageValidator()

    detected_count = 0
    valid_landmarks_count = 0
    confidences = []
    aspect_ratios = []
    latencies_ms = []

    for idx, p in enumerate(img_paths):
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue

        # Benchmark inference time
        t0 = time.perf_counter()
        detections = segmenter.segment(img_bgr)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)

        if detections:
            detected_count += 1
            primary = detections[0]
            confidences.append(primary.confidence)

            # Extract landmarks
            landmarks = extractor.extract_landmarks(primary.mask)
            if landmarks is not None and landmarks.is_valid:
                valid_landmarks_count += 1
                aspect_ratios.append(landmarks.aspect_ratio)

    det_rate = (detected_count / total_samples) * 100 if total_samples > 0 else 0
    landmark_rate = (valid_landmarks_count / detected_count) * 100 if detected_count > 0 else 0
    mean_conf = float(np.mean(confidences)) if confidences else 0.0
    mean_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    median_latency = float(np.median(latencies_ms)) if latencies_ms else 0.0
    mean_ar = float(np.mean(aspect_ratios)) if aspect_ratios else 0.0

    print("-" * 65)
    print("EVALUATION METRICS SUMMARY:")
    print(f"  Total test samples evaluated:      {total_samples}")
    print(f"  Foot detection success rate:       {detected_count}/{total_samples} ({det_rate:.1f}%)")
    print(f"  Mean detection confidence:         {mean_conf * 100:.2f}%")
    print(f"  Valid anatomical landmarks rate:   {valid_landmarks_count}/{detected_count} ({landmark_rate:.1f}%)")
    print(f"  Mean foot aspect ratio (L/W):      {mean_ar:.2f} (biological range ~ 2.2 - 2.8)")
    print(f"  Mean inference latency (CPU):      {mean_latency:.1f} ms")
    print(f"  Median inference latency (CPU):    {median_latency:.1f} ms")
    print("-" * 65)
    print("Note: Absolute millimeter MAE requires ground truth millimeter caliper")
    print("measurements for each photograph. Calibration is validated independently")
    print("in unit tests with known metric ground truths.")
    print("=" * 65)

if __name__ == "__main__":
    evaluate_dataset()
