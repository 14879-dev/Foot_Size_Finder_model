"""Foot landmark extraction module.

Extracts anatomical heel and toe endpoints, the longitudinal foot axis,
and cross-sectional foot width from cleaned binary segmentation masks.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np

@dataclass
class FootLandmarks:
    """Contains anatomical landmark points, axes, and measurements in pixel space."""
    heel_point: Tuple[float, float]     # (x, y) coordinates of heel center
    toe_point: Tuple[float, float]      # (x, y) coordinates of longest toe tip
    length_pixels: float                # Euclidean distance between heel and toe
    width_pixels: float                 # Maximum foot width perpendicular to longitudinal axis
    aspect_ratio: float                 # Length to width ratio
    orientation_deg: float              # Angle of foot vector in degrees (-180 to 180)
    center_of_mass: Tuple[float, float] # (cx, cy)
    is_valid: bool                      # Whether geometric heuristics pass
    quality_notes: str                  # Diagnostic notes or warnings


class FootLandmarkExtractor:
    """Extracts heel-to-toe longitudinal axis and endpoints using PCA and morphological width gradients."""

    def __init__(
        self,
        morph_kernel_size: int = 7,
        min_aspect_ratio: float = 1.6,
        max_aspect_ratio: float = 4.2
    ):
        self.morph_kernel_size = morph_kernel_size
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Applies morphological closing and opening to eliminate noise, holes, and edge fraying."""
        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size)
        )
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
        return opened

    def extract_landmarks(
        self,
        mask: np.ndarray,
        expected_toe_direction: Optional[str] = None # 'up', 'down', 'left', 'right', or None for auto
    ) -> Optional[FootLandmarks]:
        """Extracts heel and toe endpoints from a binary foot segmentation mask.

        Args:
            mask: 2D binary uint8 mask [H, W] (values 0 and 255).
            expected_toe_direction: Optional hint ('up', 'down', 'left', 'right', or None).

        Returns:
            FootLandmarks object or None if no valid foot contour could be processed.
        """
        if mask is None or np.sum(mask > 0) < 100:
            return None

        cleaned = self.clean_mask(mask)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 200:
            return None

        # Extract (x, y) coordinates of contour points
        pts = largest_contour.reshape(-1, 2).astype(np.float32)

        # 1. Compute Center of Mass and Principal Component Analysis (PCA)
        mean, eigenvectors = cv2.PCACompute(pts, mean=None)
        center = tuple(mean[0])  # (cx, cy)
        v_longitudinal = eigenvectors[0] # Primary axis [vx, vy]
        v_perpendicular = eigenvectors[1] # Transverse axis [wx, wy]

        # Normalize direction vectors
        v_longitudinal = v_longitudinal / np.linalg.norm(v_longitudinal)
        v_perpendicular = v_perpendicular / np.linalg.norm(v_perpendicular)

        # 2. Project all contour points onto longitudinal axis
        shifted_pts = pts - mean
        proj_long = np.dot(shifted_pts, v_longitudinal)
        proj_perp = np.dot(shifted_pts, v_perpendicular)

        min_idx = np.argmin(proj_long)
        max_idx = np.argmax(proj_long)

        cand_a = pts[min_idx] # Extreme point in negative v direction
        cand_b = pts[max_idx] # Extreme point in positive v direction

        # 3. Determine which end is Toe vs Heel
        # Anatomical rule: Forefoot/toe region is wider than the calcaneus/heel region.
        # Sample points in the 15% - 30% range from end A and end B along longitudinal axis
        range_span = proj_long.max() - proj_long.min()
        band_size = 0.20 * range_span

        # Band A (near cand_a)
        band_a_mask = (proj_long >= proj_long.min() + 0.05 * range_span) & \
                      (proj_long <= proj_long.min() + 0.05 * range_span + band_size)
        # Band B (near cand_b)
        band_b_mask = (proj_long <= proj_long.max() - 0.05 * range_span) & \
                      (proj_long >= proj_long.max() - 0.05 * range_span - band_size)

        width_a = (proj_perp[band_a_mask].max() - proj_perp[band_a_mask].min()) if np.any(band_a_mask) else 0
        width_b = (proj_perp[band_b_mask].max() - proj_perp[band_b_mask].min()) if np.any(band_b_mask) else 0

        # Refine endpoints: Average top 1.5% furthest points to suppress single-pixel spike artifacts
        num_refine = max(3, int(len(pts) * 0.015))
        sorted_indices = np.argsort(proj_long)
        toe_candidates_a = pts[sorted_indices[:num_refine]]
        toe_candidates_b = pts[sorted_indices[-num_refine:]]

        avg_a = np.mean(toe_candidates_a, axis=0)
        avg_b = np.mean(toe_candidates_b, axis=0)

        # Check user hint if available, otherwise use width asymmetry
        if expected_toe_direction == "up":
            # Toe is higher in image (smaller y)
            is_b_toe = avg_b[1] < avg_a[1]
        elif expected_toe_direction == "down":
            is_b_toe = avg_b[1] > avg_a[1]
        elif expected_toe_direction == "left":
            is_b_toe = avg_b[0] < avg_a[0]
        elif expected_toe_direction == "right":
            is_b_toe = avg_b[0] > avg_a[0]
        else:
            # Auto: Forefoot is wider than heel
            is_b_toe = width_b >= width_a

        if is_b_toe:
            toe_pt = (float(avg_b[0]), float(avg_b[1]))
            heel_pt = (float(avg_a[0]), float(avg_a[1]))
        else:
            toe_pt = (float(avg_a[0]), float(avg_a[1]))
            heel_pt = (float(avg_b[0]), float(avg_b[1]))

        # Calculate pixel dimensions
        dx = toe_pt[0] - heel_pt[0]
        dy = toe_pt[1] - heel_pt[1]
        length_px = float(np.sqrt(dx * dx + dy * dy))

        # Cross-sectional width: maximum span across perpendicular axis
        width_px = float(proj_perp.max() - proj_perp.min())

        aspect_ratio = (length_px / width_px) if width_px > 0 else 0.0
        angle_deg = float(np.degrees(np.arctan2(dy, dx)))

        # Validation heuristics
        notes = []
        is_valid = True

        if aspect_ratio < self.min_aspect_ratio:
            is_valid = False
            notes.append(f"Foot aspect ratio {aspect_ratio:.2f} is unusually round (expected > {self.min_aspect_ratio}).")
        elif aspect_ratio > self.max_aspect_ratio:
            is_valid = False
            notes.append(f"Foot aspect ratio {aspect_ratio:.2f} is excessively narrow (expected < {self.max_aspect_ratio}).")

        # Check if points touch boundary (truncated foot)
        h, w = mask.shape[:2]
        margin = 4
        for pt_name, pt in [("Toe", toe_pt), ("Heel", heel_pt)]:
            if pt[0] <= margin or pt[0] >= w - margin or pt[1] <= margin or pt[1] >= h - margin:
                notes.append(f"{pt_name} touches image boundary (potential truncation).")
                is_valid = False

        quality_notes = "; ".join(notes) if notes else "Landmarks verified within normal anatomical bounds."

        return FootLandmarks(
            heel_point=heel_pt,
            toe_point=toe_pt,
            length_pixels=length_px,
            width_pixels=width_px,
            aspect_ratio=aspect_ratio,
            orientation_deg=angle_deg,
            center_of_mass=(float(center[0]), float(center[1])),
            is_valid=is_valid,
            quality_notes=quality_notes
        )

    @staticmethod
    def draw_landmarks(
        image_bgr: np.ndarray,
        landmarks: FootLandmarks,
        draw_axis: bool = True
    ) -> np.ndarray:
        """Annotates image with heel/toe endpoints, longitudinal measurement line, and orientation."""
        output = image_bgr.copy()
        heel = (int(round(landmarks.heel_point[0])), int(round(landmarks.heel_point[1])))
        toe = (int(round(landmarks.toe_point[0])), int(round(landmarks.toe_point[1])))

        if draw_axis:
            # Longitudinal measurement line (Heel -> Toe)
            cv2.line(output, heel, toe, (0, 255, 0), 2, cv2.LINE_AA)

        # Heel marker (Deep Amber/Orange circle)
        cv2.circle(output, heel, 7, (0, 140, 255), -1, cv2.LINE_AA)
        cv2.circle(output, heel, 9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            output, "Heel", (heel[0] + 12, heel[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2, cv2.LINE_AA
        )

        # Toe marker (Bright Cyan/Blue circle)
        cv2.circle(output, toe, 7, (255, 100, 0), -1, cv2.LINE_AA)
        cv2.circle(output, toe, 9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            output, "Toe Tip", (toe[0] + 12, toe[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 0), 2, cv2.LINE_AA
        )

        # Display pixel length tag at midpoint
        midpoint = ((heel[0] + toe[0]) // 2, (heel[1] + toe[1]) // 2)
        dist_tag = f"{landmarks.length_pixels:.1f} px"
        cv2.putText(
            output, dist_tag, (midpoint[0] + 10, midpoint[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA
        )
        return output
