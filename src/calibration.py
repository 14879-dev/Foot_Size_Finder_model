"""Reference object calibration module.

Detects reference objects (e.g. standard ISO/IEC 7810 ID-1 cards or A4 paper),
computes perspective homography transforms, and calculates pixels-per-millimeter (PPM).
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
import cv2
import numpy as np

# Standard dimensions in millimeters
STANDARD_CARD_WIDTH_MM = 85.60
STANDARD_CARD_HEIGHT_MM = 53.98
STANDARD_CARD_ASPECT_RATIO = STANDARD_CARD_WIDTH_MM / STANDARD_CARD_HEIGHT_MM  # ~1.5858

A4_WIDTH_MM = 297.0
A4_HEIGHT_MM = 210.0
A4_ASPECT_RATIO = A4_WIDTH_MM / A4_HEIGHT_MM  # ~1.4142

@dataclass
class CalibrationData:
    """Contains calibration scale, homography transform, and quality metrics."""
    pixels_per_mm: float
    reference_object: str
    reference_dims_mm: Tuple[float, float]     # (width_mm, height_mm)
    card_corners: Optional[np.ndarray]         # 4 ordered corners [[x,y],...]
    homography_matrix: Optional[np.ndarray]    # 3x3 matrix mapping image coords -> mm coords
    is_perspective_corrected: bool
    calibration_quality: str                   # 'High', 'Moderate', 'Low', 'Failed'
    quality_notes: str


class ReferenceCardCalibrator:
    """Calibrates 2D planar measurements using a standard reference card or document."""

    def __init__(
        self,
        card_width_mm: float = STANDARD_CARD_WIDTH_MM,
        card_height_mm: float = STANDARD_CARD_HEIGHT_MM,
        aspect_ratio_tolerance: float = 0.35
    ):
        self.card_width_mm = card_width_mm
        self.card_height_mm = card_height_mm
        self.target_aspect_ratio = max(card_width_mm, card_height_mm) / min(card_width_mm, card_height_mm)
        self.aspect_ratio_tolerance = aspect_ratio_tolerance

    @staticmethod
    def order_corners(pts: np.ndarray) -> np.ndarray:
        """Orders 4 points as: [top-left, top-right, bottom-right, bottom-left]."""
        pts = pts.reshape(4, 2).astype(np.float32)
        rect = np.zeros((4, 2), dtype=np.float32)

        # Top-left has smallest sum (x + y), bottom-right has largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Top-right has smallest difference (y - x), bottom-left has largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def auto_detect_card(
        self,
        image_bgr: np.ndarray,
        exclusion_mask: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Robustly detects a rectangular reference card (or A4 paper) in the image.

        Uses multi-strategy edge and threshold extraction, verifies 4 convex corners,
        enforces ~90 degree internal angles, matches target aspect ratio, and strictly
        rejects any candidate that overlaps with the foot mask.

        Args:
            image_bgr: Input color image.
            exclusion_mask: Foot binary mask (0 or 255) to strictly exclude.

        Returns:
            Ordered 4x2 array of corner points [TL, TR, BR, BL], or None if not found.
        """
        h, w = image_bgr.shape[:2]
        img_area = h * w
        min_card_area = img_area * 0.008   # At least 0.8% of image
        max_card_area = img_area * 0.45    # At most 45% of image

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Multi-strategy binary edge maps
        binary_maps = []

        # Strategy 1: Adaptive Thresholding (great for varying floor lighting)
        th_adapt1 = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 4
        )
        binary_maps.append(th_adapt1)
        binary_maps.append(cv2.bitwise_not(th_adapt1))

        # Strategy 2: Multi-scale Canny edge detection with closing
        for low, high in [(30, 100), (50, 150), (80, 200)]:
            edges = cv2.Canny(blurred, low, high)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(edges, kernel, iterations=1)
            binary_maps.append(dilated)

        # Strategy 3: Otsu threshold on saturation channel (cards often have distinct saturation)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        _, th_sat = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_maps.append(th_sat)

        candidates = []

        for b_map in binary_maps:
            contours, _ = cv2.findContours(b_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_card_area or area > max_card_area:
                    continue

                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue

                # Test multiple epsilon values to find clean 4-corner polygon
                for eps_factor in [0.02, 0.03, 0.04]:
                    approx = cv2.approxPolyDP(cnt, eps_factor * perimeter, True)

                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        pts = approx.reshape(4, 2).astype(np.float32)
                        ordered = self.order_corners(pts)
                        tl, tr, br, bl = ordered

                        # 1. Overlap Check: Reject candidate if it overlaps with foot
                        if exclusion_mask is not None:
                            card_mask = np.zeros((h, w), dtype=np.uint8)
                            cv2.fillPoly(card_mask, [ordered.astype(np.int32)], 255)
                            foot_overlap = np.sum((card_mask > 0) & (exclusion_mask > 0))
                            if foot_overlap / max(1, area) > 0.05:
                                continue

                        # 2. Side length analysis
                        w_top = np.linalg.norm(tr - tl)
                        w_bottom = np.linalg.norm(br - bl)
                        h_left = np.linalg.norm(bl - tl)
                        h_right = np.linalg.norm(br - tr)

                        avg_w = (w_top + w_bottom) / 2.0
                        avg_h = (h_left + h_right) / 2.0
                        if avg_w == 0 or avg_h == 0:
                            continue

                        # Opposing edge parallelism check (card shouldn't be excessively trapezoidal)
                        w_diff = abs(w_top - w_bottom) / max(w_top, w_bottom)
                        h_diff = abs(h_left - h_right) / max(h_left, h_right)
                        if w_diff > 0.35 or h_diff > 0.35:
                            continue

                        # 3. Aspect ratio check
                        observed_ratio = max(avg_w, avg_h) / min(avg_w, avg_h)
                        ratio_diff = abs(observed_ratio - self.target_aspect_ratio)
                        if ratio_diff > self.aspect_ratio_tolerance:
                            continue

                        # 4. Angle check: All 4 internal angles must be roughly perpendicular (~90 deg)
                        edges = [tr - tl, br - tr, bl - br, tl - bl]
                        angles = []
                        valid_angles = True
                        for i in range(4):
                            v1 = edges[i]
                            v2 = -edges[(i + 3) % 4]
                            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                            if angle_deg < 65.0 or angle_deg > 115.0:
                                valid_angles = False
                                break
                            angles.append(abs(angle_deg - 90.0))

                        if not valid_angles:
                            continue

                        # 5. Rectangularity / solidity check
                        rect = cv2.minAreaRect(cnt)
                        box_area = rect[1][0] * rect[1][1]
                        solidity = area / max(1.0, box_area)
                        if solidity < 0.82:
                            continue

                        # Composite score: lower is better (ratio difference + angle deviations - size reward)
                        angle_penalty = np.mean(angles) / 10.0
                        score = ratio_diff * 2.0 + angle_penalty + (w_diff + h_diff)
                        candidates.append((score, area, ordered))

        if not candidates:
            return None

        # Sort by best score (lowest deviation from ideal geometry)
        candidates.sort(key=lambda c: c[0])
        return candidates[0][2]

    def calibrate_from_corners(
        self,
        corners: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> CalibrationData:
        """Computes calibration scale and perspective homography from 4 ordered card corners.

        Args:
            corners: 4x2 array of corner points [tl, tr, br, bl].
            image_shape: (height, width) of source image.

        Returns:
            CalibrationData instance.
        """
        ordered = self.order_corners(corners)
        tl, tr, br, bl = ordered

        w_top = np.linalg.norm(tr - tl)
        w_bottom = np.linalg.norm(br - bl)
        h_left = np.linalg.norm(bl - tl)
        h_right = np.linalg.norm(br - tr)

        avg_w = (w_top + w_bottom) / 2.0
        avg_h = (h_left + h_right) / 2.0

        # Determine if card is oriented horizontally (width > height) or vertically
        if avg_w >= avg_h:
            target_w_mm = max(self.card_width_mm, self.card_height_mm)
            target_h_mm = min(self.card_width_mm, self.card_height_mm)
        else:
            target_w_mm = min(self.card_width_mm, self.card_height_mm)
            target_h_mm = max(self.card_width_mm, self.card_height_mm)

        # Standard canonical destination points in millimeters
        dst_pts_mm = np.array([
            [0.0, 0.0],
            [target_w_mm, 0.0],
            [target_w_mm, target_h_mm],
            [0.0, target_h_mm]
        ], dtype=np.float32)

        # Compute Homography: Image Plane -> Real-world Millimeter Plane
        homography_matrix, _ = cv2.findHomography(ordered, dst_pts_mm)

        # Simple diagonal scale for PPM
        diag_px = float(np.linalg.norm(br - tl))
        diag_mm = float(np.sqrt(target_w_mm**2 + target_h_mm**2))
        ppm = diag_px / diag_mm if diag_mm > 0 else 1.0

        # Quality assessment
        notes = []
        quality = "High"

        # Check card size in pixels
        if diag_px < 100.0:
            quality = "Low"
            notes.append("Card appears very small or distant; measurement resolution is limited.")
        elif diag_px < 180.0:
            quality = "Moderate"
            notes.append("Card size is moderate; good accuracy expected.")
        else:
            notes.append("Card resolution is crisp and well-scaled.")

        # Check perspective skew (parallelism between opposing edges)
        w_ratio = abs(w_top - w_bottom) / max(w_top, w_bottom)
        h_ratio = abs(h_left - h_right) / max(h_left, h_right)
        if w_ratio > 0.30 or h_ratio > 0.30:
            notes.append("Card is viewed at a steep perspective angle; 2D homography applied.")
            if quality == "High":
                quality = "Moderate"

        return CalibrationData(
            pixels_per_mm=ppm,
            reference_object="Standard Reference Card",
            reference_dims_mm=(target_w_mm, target_h_mm),
            card_corners=ordered,
            homography_matrix=homography_matrix,
            is_perspective_corrected=(homography_matrix is not None),
            calibration_quality=quality,
            quality_notes="; ".join(notes)
        )

    def calibrate(
        self,
        image_bgr: np.ndarray,
        manual_corners: Optional[Union[np.ndarray, List]] = None,
        exclusion_mask: Optional[np.ndarray] = None
    ) -> Optional[CalibrationData]:
        """Performs full calibration either from provided corners or automatic detection."""
        if manual_corners is not None and len(manual_corners) == 4:
            pts = np.array(manual_corners, dtype=np.float32)
            return self.calibrate_from_corners(pts, image_bgr.shape[:2])

        detected = self.auto_detect_card(image_bgr, exclusion_mask=exclusion_mask)
        if detected is not None:
            return self.calibrate_from_corners(detected, image_bgr.shape[:2])

        return None

    @staticmethod
    def transform_points(pts: np.ndarray, homography_matrix: np.ndarray) -> np.ndarray:
        """Applies homography to points from image space to calibrated millimeter space."""
        pts_homo = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
        transformed = np.dot(homography_matrix, pts_homo.T).T
        # Normalize by scale factor w
        w = transformed[:, 2:3]
        w[w == 0] = 1e-6
        return transformed[:, 0:2] / w

    @staticmethod
    def draw_calibration(
        image_bgr: np.ndarray,
        calibration: CalibrationData
    ) -> np.ndarray:
        """Annotates image with detected reference card polygon and corner markers."""
        output = image_bgr.copy()
        if calibration.card_corners is None:
            return output

        corners = calibration.card_corners.astype(int)

        # Draw card polygon boundary (Emerald green)
        cv2.polylines(output, [corners], isClosed=True, color=(0, 220, 100), thickness=2, lineType=cv2.LINE_AA)

        # Corner labels
        labels = ["TL", "TR", "BR", "BL"]
        colors = [(0, 255, 255), (0, 200, 255), (0, 150, 255), (0, 100, 255)]

        for pt, lbl, col in zip(corners, labels, colors):
            cv2.circle(output, tuple(pt), 6, col, -1, cv2.LINE_AA)
            cv2.circle(output, tuple(pt), 8, (255, 255, 255), 1, cv2.LINE_AA)

        # Tag
        top_center = (int((corners[0][0] + corners[1][0]) / 2), int((corners[0][1] + corners[1][1]) / 2))
        tag = f"Ref Card: {calibration.pixels_per_mm:.2f} px/mm"
        cv2.putText(
            output,
            tag,
            (top_center[0] - 60, max(25, top_center[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 100),
            2,
            cv2.LINE_AA
        )
        return output
