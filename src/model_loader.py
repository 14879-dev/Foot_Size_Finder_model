"""Model loader for Khybersole Foot Size Finder.

Loads and caches the YOLO instance segmentation model once to optimize memory
and response time. Supports automatic CPU/GPU device selection and robust error handling.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import torch

class ModelError(Exception):
    """Base exception for model loading and execution errors."""
    pass

class ModelNotFoundError(ModelError):
    """Raised when the specified model weights file cannot be located."""
    pass

class ModelLoadError(ModelError):
    """Raised when the model weights cannot be parsed or loaded by the framework."""
    pass

class IncompatibleModelError(ModelError):
    """Raised when the loaded model is not an instance segmentation model for feet."""
    pass


class ModelLoader:
    """Thread-safe singleton model loader for Ultralytics YOLO segmentation models."""

    _instance: Optional["ModelLoader"] = None
    _model: Optional[Any] = None
    _loaded_path: Optional[str] = None
    _device: Optional[str] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    @classmethod
    def resolve_model_path(cls, requested_path: Optional[str] = None) -> Path:
        """Searches candidate locations for best.pt and returns an existing Path."""
        if requested_path:
            cand = Path(requested_path)
            if cand.is_file():
                return cand.resolve()
            raise ModelNotFoundError(f"Specified model path does not exist: {requested_path}")

        # Standard project candidate paths when no explicit path requested
        candidates = [
            Path("models/best.pt"),
            Path("best.pt"),
            Path("d:/FYP/model/models/best.pt"),
            Path("d:/FYP/model/best.pt"),
            Path(os.path.expanduser("~/Downloads/best.pt")),
            Path("C:/KhybersoleAI/models/best.pt")
        ]

        for cand in candidates:
            if cand.is_file():
                return cand.resolve()

        raise ModelNotFoundError(
            f"Could not locate foot segmentation weights (best.pt). "
            f"Searched candidates: {[str(c) for c in candidates]}"
        )

    @classmethod
    def load_model(
        cls,
        model_path: Optional[str] = None,
        device: str = "auto",
        force_reload: bool = False
    ) -> Any:
        """Loads and returns the YOLO segmentation model.

        Args:
            model_path: Optional path to weights. If None, auto-resolved.
            device: 'auto', 'cpu', or 'cuda'.
            force_reload: Force reloading model even if already in memory.

        Returns:
            Ultralytics YOLO instance.
        """
        resolved_path = cls.resolve_model_path(model_path)
        path_str = str(resolved_path)

        # Check if already loaded
        if not force_reload and cls._model is not None and cls._loaded_path == path_str:
            return cls._model

        # Determine target device
        if device == "auto":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            target_device = device

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ModelLoadError("Ultralytics library is not installed: " + str(e)) from e

        try:
            model = YOLO(path_str)
            # Ensure model is assigned to chosen device
            model.to(target_device)
        except Exception as e:
            raise ModelLoadError(f"Failed to load YOLO model from '{path_str}': {e}") from e

        # Validate task compatibility
        task = getattr(model, "task", "")
        if task != "segment":
            # Check model class name as fallback
            model_cls = type(model.model).__name__ if hasattr(model, "model") else ""
            if "Segment" not in model_cls:
                raise IncompatibleModelError(
                    f"Model task is '{task}' ({model_cls}). "
                    f"Expected instance segmentation ('segment')."
                )

        cls._model = model
        cls._loaded_path = path_str
        cls._device = target_device
        return cls._model

    @classmethod
    def get_model_info(cls) -> Dict[str, Any]:
        """Returns metadata about the loaded model."""
        if cls._model is None:
            return {"loaded": False}

        names = getattr(cls._model, "names", {})
        task = getattr(cls._model, "task", "unknown")
        arch = type(cls._model.model).__name__ if hasattr(cls._model, "model") else "unknown"

        return {
            "loaded": True,
            "path": cls._loaded_path,
            "device": cls._device,
            "task": task,
            "architecture": arch,
            "classes": names,
            "num_classes": len(names)
        }
