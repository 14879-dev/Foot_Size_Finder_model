"""Foot segmentation module using Ultralytics YOLOv8 segmentation model.

Extracts foot masks, bounding boxes, and confidence scores from photographic inputs,
scaling masks back to original image coordinates.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from src.model_loader import ModelLoader, ModelError

@dataclass
class FootDetectionResult:
    """Represents a single foot detection and segmentation outcome."""
    mask: np.ndarray             # Binary mask [H, W] of uint8 (0 or 255)
    bbox: Tuple[int, int, int, int] # (x1, y1, x2, y2)
    confidence: float            # Detection confidence [0.0 - 1.0]
    class_name: str              # Detected class name ('foot')
    area_pixels: int             # Total white pixels in mask
    contour: np.ndarray          # Largest external contour points

class FootSegmenter:
    """Performs foot segmentation inference and extracts polygon/raster masks."""

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        self.model = ModelLoader.load_model(model_path)

    @staticmethod
    def _prepare_image(image_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Converts diverse image input formats into a BGR OpenCV NumPy array."""
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Could not read image from file: {image_input}")
            return img
        elif isinstance(image_input, Image.Image):
            # PIL image -> BGR
            rgb = np.array(image_input.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
            return image_input.copy()
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

    def segment(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        conf_threshold: Optional[float] = None
    ) -> List[FootDetectionResult]:
        """Segments feet in the image and returns a list of FootDetectionResults.

        Args:
            image_input: Image path, ndarray, or PIL Image.
            conf_threshold: Override confidence threshold.

        Returns:
            List of FootDetectionResult, sorted descending by confidence.
        """
        img_bgr = self._prepare_image(image_input)
        orig_h, orig_w = img_bgr.shape[:2]
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold

        results = self.model(img_bgr, conf=threshold, verbose=False)
        if not results:
            return []

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []

        detections = []
        has_masks = res.masks is not None and len(res.masks) > 0

        for i, box in enumerate(res.boxes):
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = self.model.names.get(cls_id, "foot")

            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)

            if has_masks and i < len(res.masks.data):
                # Mask tensor in YOLO is typically [640, 640] or matching inference scale
                raw_mask = res.masks.data[i].cpu().numpy().astype(np.float32)
                # Resize directly to original image dimensions [orig_h, orig_w]
                mask_resized = cv2.resize(
                    raw_mask,
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_LINEAR
                )
                bin_mask = (mask_resized > 0.5).astype(np.uint8) * 255
            else:
                # Fallback to rectangle mask if model lacks segmentation mask
                bin_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                bin_mask[y1:y2, x1:x2] = 255

            # Find largest external contour of this mask
            contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
            else:
                largest_contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]])

            area = int(np.sum(bin_mask > 0))

            detections.append(
                FootDetectionResult(
                    mask=bin_mask,
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_name=cls_name,
                    area_pixels=area,
                    contour=largest_contour
                )
            )

        # Sort detections by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def get_primary_foot(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        conf_threshold: Optional[float] = None
    ) -> Optional[FootDetectionResult]:
        """Returns the single most prominent detected foot, or None if no foot detected."""
        detections = self.segment(image_input, conf_threshold=conf_threshold)
        if not detections:
            return None
        # Return detection with highest confidence (or largest area if multiple high confidence)
        return detections[0]

    @staticmethod
    def draw_overlay(
        image_bgr: np.ndarray,
        result: FootDetectionResult,
        color_bgr: Tuple[int, int, int] = (0, 200, 255),
        alpha: float = 0.4
    ) -> np.ndarray:
        """Draws an elegant semi-transparent segmentation overlay and contour."""
        overlay = image_bgr.copy()
        mask_bool = result.mask > 0

        # Apply colored tint over the mask region
        colored_mask = np.zeros_like(image_bgr)
        colored_mask[mask_bool] = color_bgr
        cv2.addWeighted(colored_mask, alpha, overlay, 1.0 - alpha, 0, overlay)

        # Blend only over the masked region
        output = image_bgr.copy()
        output[mask_bool] = overlay[mask_bool]

        # Draw contour line for sharp perimeter
        cv2.drawContours(output, [result.contour], -1, color_bgr, 2, cv2.LINE_AA)

        # Draw bounding box
        x1, y1, x2, y2 = result.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)

        # Add label tag
        tag = f"Foot: {result.confidence * 100:.1f}%"
        cv2.putText(
            output,
            tag,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        return output
