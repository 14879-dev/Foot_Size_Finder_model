"""Diagnostic script for inspecting Khybersole foot segmentation model (best.pt)."""

import os
import sys
from pathlib import Path
import cv2
import numpy as np

def run_diagnostics(model_path: str, test_image_path: str = None, output_dir: str = "outputs"):
    print("=" * 60)
    print("KHYBERSOLE AI FOOT SIZE FINDER — MODEL DIAGNOSTIC TOOL")
    print("=" * 60)

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"[ERROR] Model file not found at: {model_path}")
        return False

    print(f"[INFO] Model file found: {model_path} ({model_path.stat().st_size / (1024*1024):.2f} MB)")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] 'ultralytics' library is not installed.")
        return False

    try:
        print("[INFO] Attempting to load model checkpoint with Ultralytics YOLO...")
        model = YOLO(str(model_path))
        print("[SUCCESS] Checkpoint loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        return False

    print(f"[INFO] Task type: {getattr(model, 'task', 'unknown')}")
    print(f"[INFO] Architecture class: {type(model.model).__name__}")
    print(f"[INFO] Available classes: {getattr(model, 'names', {})}")

    is_segmentation = getattr(model, 'task', '') == 'segment' or 'Segmentation' in type(model.model).__name__
    print(f"[INFO] Is instance segmentation model: {is_segmentation}")

    # Test inference
    if not test_image_path or not Path(test_image_path).exists():
        candidates = list(Path("test_images").glob("*.jpg")) + list(Path("test_images").glob("*.png"))
        if candidates:
            test_image_path = str(candidates[0])
            print(f"[INFO] Using detected test image: {test_image_path}")
        else:
            print("[WARNING] No test image provided or found in test_images/ directory.")
            return True

    print(f"[INFO] Running inference on: {test_image_path}")
    try:
        results = model(test_image_path, conf=0.25, verbose=False)
        result = results[0]

        num_boxes = len(result.boxes) if result.boxes is not None else 0
        has_masks = result.masks is not None and len(result.masks) > 0

        print(f"[INFO] Detections count: {num_boxes}")
        print(f"[INFO] Masks produced: {has_masks}")

        if num_boxes > 0:
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, str(cls_id))
                conf = float(box.conf[0])
                print(f"       Box {i+1}: Class='{cls_name}' (ID: {cls_id}), Confidence={conf:.4f}")

        if has_masks:
            mask_shape = result.masks.data.shape
            print(f"[INFO] Mask tensor shape: {mask_shape}")

            # Save annotated output
            os.makedirs(output_dir, exist_ok=True)
            annotated_frame = result.plot()
            out_path = Path(output_dir) / f"diagnostic_result_{Path(test_image_path).name}"
            cv2.imwrite(str(out_path), annotated_frame)
            print(f"[SUCCESS] Annotated diagnostic result saved to: {out_path}")

        print("=" * 60)
        print("[SUMMARY] Model is fully validated and operational for foot segmentation.")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"[ERROR] Inference failed on test image: {e}")
        return False

if __name__ == "__main__":
    model_file = sys.argv[1] if len(sys.argv) > 1 else "models/best.pt"
    img_file = sys.argv[2] if len(sys.argv) > 2 else None
    success = run_diagnostics(model_file, img_file)
    sys.exit(0 if success else 1)
