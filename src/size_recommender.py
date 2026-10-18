"""Peshawari Chappal size recommendation engine.

Maps measured foot length in millimeters to traditional Peshawari chappal sizes (PK/UK, EU, US)
based on an external JSON configuration file, with boundary analysis and fit allowances.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

@dataclass
class SizeRecommendationResult:
    """Detailed size recommendation outcome."""
    foot_length_mm: float
    foot_length_cm: float
    recommended_size: Optional[int]
    label: str
    size_eu: Optional[int]
    size_us: Optional[int]
    fit_type: str                     # 'Standard Fit', 'Snug Fit', 'Roomy / Wide Fit'
    boundary_warning: Optional[str]   # If measurement is close to transition border
    alternative_size: Optional[int]   # Neighboring size if near boundary
    is_covered: bool                  # Whether foot length is within chart coverage
    notes: str
    chart_status: str                 # 'sample_development_data' or 'verified_manufacturer_data'


class SizeRecommender:
    """Configurable sizing engine for Khybersole footwear."""

    def __init__(self, chart_path: Optional[str] = None):
        self.chart_path = Path(chart_path) if chart_path else Path("config/size_chart.json")
        self.data: Dict[str, Any] = {}
        self.sizes: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self.load_chart()

    def load_chart(self, new_path: Optional[str] = None) -> None:
        """Loads or reloads the size chart from disk."""
        target_path = Path(new_path) if new_path else self.chart_path
        if not target_path.exists():
            # Fallback path if running from subfolder
            alt_path = Path("../config/size_chart.json")
            if alt_path.exists():
                target_path = alt_path
            else:
                raise FileNotFoundError(f"Size chart file not found at: {target_path}")

        with open(target_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.metadata = self.data.get("chart_metadata", {})
        self.sizes = self.data.get("sizes", [])
        self.chart_path = target_path

    def recommend_size(self, foot_length_mm: float) -> SizeRecommendationResult:
        """Finds recommended Peshawari chappal size for a given foot length in millimeters.

        Args:
            foot_length_mm: Measured anatomical foot length.

        Returns:
            SizeRecommendationResult with size, fit details, and boundary flags.
        """
        foot_length_cm = round(foot_length_mm / 10.0, 1)
        status = self.metadata.get("status", "sample_development_data")

        if not self.sizes:
            return SizeRecommendationResult(
                foot_length_mm=foot_length_mm,
                foot_length_cm=foot_length_cm,
                recommended_size=None,
                label="No Size Configured",
                size_eu=None,
                size_us=None,
                fit_type="Unknown",
                boundary_warning=None,
                alternative_size=None,
                is_covered=False,
                notes="Size chart is empty or could not be parsed.",
                chart_status=status
            )

        # Min and Max supported across all sizes
        min_supported = min(s["min_foot_length_mm"] for s in self.sizes)
        max_supported = max(s["max_foot_length_mm"] for s in self.sizes)

        # Out-of-bounds checks
        if foot_length_mm < min_supported:
            smallest = min(self.sizes, key=lambda s: s["min_foot_length_mm"])
            return SizeRecommendationResult(
                foot_length_mm=foot_length_mm,
                foot_length_cm=foot_length_cm,
                recommended_size=None,
                label=f"Below Minimum Size ({smallest['label']})",
                size_eu=None,
                size_us=None,
                fit_type="Below Range",
                boundary_warning=f"Foot length {foot_length_cm} cm is smaller than our smallest standard size ({min_supported/10:.1f} cm).",
                alternative_size=smallest["size_pk_uk"],
                is_covered=False,
                notes="Custom artisan sizing is recommended for petite feet.",
                chart_status=status
            )

        if foot_length_mm > max_supported:
            largest = max(self.sizes, key=lambda s: s["max_foot_length_mm"])
            return SizeRecommendationResult(
                foot_length_mm=foot_length_mm,
                foot_length_cm=foot_length_cm,
                recommended_size=None,
                label=f"Above Maximum Size ({largest['label']})",
                size_eu=None,
                size_us=None,
                fit_type="Above Range",
                boundary_warning=f"Foot length {foot_length_cm} cm exceeds standard catalogue sizes (max {max_supported/10:.1f} cm).",
                alternative_size=largest["size_pk_uk"],
                is_covered=False,
                notes="Special custom Peshawari craftsman order required.",
                chart_status=status
            )

        # Iterate over size intervals
        matched_size = None
        for s in self.sizes:
            if s["min_foot_length_mm"] <= foot_length_mm <= s["max_foot_length_mm"]:
                matched_size = s
                break

        # If not matched directly (e.g. between small gap), find closest
        if matched_size is None:
            matched_size = min(
                self.sizes,
                key=lambda s: min(abs(foot_length_mm - s["min_foot_length_mm"]), abs(foot_length_mm - s["max_foot_length_mm"]))
            )

        size_num = matched_size["size_pk_uk"]
        label = matched_size.get("label", f"Size {size_num}")
        eu = matched_size.get("size_eu")
        us = matched_size.get("size_us")
        recommended_len = matched_size.get("recommended_foot_length_mm", (matched_size["min_foot_length_mm"] + matched_size["max_foot_length_mm"]) / 2.0)

        # Boundary analysis: Check if within 2.0 mm of edge
        boundary_warning = None
        alt_size = None
        dist_to_min = foot_length_mm - matched_size["min_foot_length_mm"]
        dist_to_max = matched_size["max_foot_length_mm"] - foot_length_mm

        if dist_to_min < 2.0:
            # Near lower boundary
            smaller_candidates = [s["size_pk_uk"] for s in self.sizes if s["size_pk_uk"] < size_num]
            if smaller_candidates:
                alt_size = max(smaller_candidates)
                boundary_warning = f"Near size boundary (close to Size {alt_size}). If you prefer a snug fit, consider Size {alt_size}."
        elif dist_to_max < 2.0:
            # Near upper boundary
            larger_candidates = [s["size_pk_uk"] for s in self.sizes if s["size_pk_uk"] > size_num]
            if larger_candidates:
                alt_size = min(larger_candidates)
                boundary_warning = f"Near size boundary (close to Size {alt_size}). If you have wide feet or prefer extra room, consider Size {alt_size}."

        # Fit categorization
        diff_from_center = foot_length_mm - recommended_len
        if diff_from_center > 2.5:
            fit_type = "Roomy / Snug Upper Threshold"
        elif diff_from_center < -2.5:
            fit_type = "Relaxed / Generous Fit"
        else:
            fit_type = "Standard True-to-Size"

        notes = (
            f"Traditional Peshawari chappals feature leather that softens and conforms to foot contours "
            f"with wear. Standard 6mm front allowance included."
        )

        return SizeRecommendationResult(
            foot_length_mm=foot_length_mm,
            foot_length_cm=foot_length_cm,
            recommended_size=size_num,
            label=label,
            size_eu=eu,
            size_us=us,
            fit_type=fit_type,
            boundary_warning=boundary_warning,
            alternative_size=alt_size,
            is_covered=True,
            notes=notes,
            chart_status=status
        )
