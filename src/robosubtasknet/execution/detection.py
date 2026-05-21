"""Object detection and 6-DOF pose-estimation interfaces for Phase 2.

Section 12.2 of the implementation plan calls for object-centric primitives
(``reach``, ``pick``, ``place``, ``pour``, ``give``) to be parameterised from
detected object pose. We therefore expose two minimal protocols:

* :class:`ObjectDetector` — 2D bounding-box detection. The reference
  implementation :class:`YOLOv8Detector` lazy-imports ``ultralytics`` so the
  rest of the package keeps zero heavy runtime dependencies.
* :class:`PoseEstimator6D` — 4×4 SE(3) pose from RGB-D crops. We ship only a
  protocol plus a deliberately conservative :class:`DummyPoseEstimator`
  fallback that returns a centred identity-rotation pose; production setups
  should slot in GraspNet-Baseline, FoundationPose, or RGB-D template
  matching as recommended in Section 12.2.

The numpy / dict interface is intentionally library-agnostic so it can be
backed by any detector (YOLO, Florence-2, OWL-ViT, …) at swap time.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Detection",
    "ObjectDetector",
    "YOLOv8Detector",
    "PoseEstimator6D",
    "DummyPoseEstimator",
]


# Convenience alias for the detection dict shape. We use a plain ``dict``
# rather than a TypedDict to stay friendly to mypy < 1.0 and to runtime
# downstream callers that simply index into the result.
Detection = dict[str, Any]
"""Detection record::

    {
        "bbox":  np.ndarray of shape (4,)  # x1, y1, x2, y2 in image pixels
        "label": str                       # class name from the detector
        "score": float                     # confidence in [0, 1]
    }
"""


# ---------------------------------------------------------------------------
# Detector protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ObjectDetector(Protocol):
    """Minimal protocol for a 2D object detector.

    Implementations consume a BGR image as produced by OpenCV (``cv2.imread``
    or ``VideoCapture.read``) and return a list of detection dicts in the
    shape described by :data:`Detection`. The protocol is intentionally light:
    we do not constrain whether the detector batches internally or runs
    one-image-at-a-time.
    """

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        ...


# ---------------------------------------------------------------------------
# YOLOv8 implementation
# ---------------------------------------------------------------------------


_ULTRALYTICS_INSTALL_HINT = (
    "ultralytics is required for YOLOv8Detector. Install with:\n"
    "    pip install ultralytics\n"
    "or, if you maintain a robot-specific extras group in pyproject.toml,\n"
    "    pip install -e .[robot]"
)


class YOLOv8Detector:
    """YOLOv8-backed :class:`ObjectDetector`.

    ``ultralytics`` is *only* imported the first time we construct the
    detector; the module remains importable on machines that do not have it
    installed. Constructing this class without ``ultralytics`` raises
    ``ImportError`` with an install hint instead of crashing at first use.

    Parameters
    ----------
    weights
        Path to a YOLOv8 weight file (``.pt``) or a model name string the
        ultralytics hub knows (e.g. ``"yolov8n.pt"``).
    conf
        Minimum confidence to keep a detection.
    iou
        Non-max-suppression IoU threshold.
    device
        Torch device string. ``None`` defers to ultralytics' default.
    classes
        Optional list of class indices to restrict outputs to.
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        device: str | None = None,
        classes: list[int] | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised at install time
            raise ImportError(_ULTRALYTICS_INSTALL_HINT) from exc

        self._YOLO = YOLO
        self.weights = weights
        self.conf = float(conf)
        self.iou = float(iou)
        self.device = device
        self.classes = classes
        # Construct the underlying model eagerly so weight download / loading
        # failures surface immediately rather than at the first ``detect``.
        self._model = YOLO(weights)

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        """Run YOLOv8 on a single BGR image and return detections.

        The ultralytics API accepts BGR numpy arrays directly, so we only
        need to forward the call and unpack ``Results``.
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        kwargs: dict[str, Any] = {
            "conf": self.conf,
            "iou": self.iou,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        if self.classes is not None:
            kwargs["classes"] = self.classes

        results = self._model.predict(image_bgr, **kwargs)
        if not results:
            return []

        # YOLO returns a list of Results, one per image. We pass a single
        # image so we only look at index 0.
        r = results[0]
        names = r.names if hasattr(r, "names") else {}
        out: list[Detection] = []
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            return out
        # boxes.xyxy and boxes.conf are torch tensors; .cpu().numpy() is safe.
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
        except AttributeError:
            # Defensive path for non-torch backends.
            xyxy = np.asarray(boxes.xyxy)
            confs = np.asarray(boxes.conf)
            clss = np.asarray(boxes.cls, dtype=int)

        for box, score, cls_idx in zip(xyxy, confs, clss):
            out.append(
                {
                    "bbox": np.asarray(box, dtype=np.float32),
                    "label": str(names.get(int(cls_idx), int(cls_idx))),
                    "score": float(score),
                }
            )
        return out


# ---------------------------------------------------------------------------
# 6-DOF pose estimator protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class PoseEstimator6D(Protocol):
    """Estimate a 4×4 SE(3) pose from an RGB-D crop.

    The interface is intentionally narrow: ``image_rgb`` is the full RGB
    frame in ``H × W × 3`` uint8, ``depth`` is an ``H × W`` float32 depth
    map in metres (zero where invalid), and ``bbox`` is a ``(4,)`` xyxy
    bounding box selecting the region of interest. Returning the full 4×4
    matrix — rather than a (translation, quaternion) tuple — keeps the
    composition with kinematic chains trivial.
    """

    def estimate(
        self,
        image_rgb: np.ndarray,
        depth: np.ndarray,
        bbox: np.ndarray,
    ) -> np.ndarray:
        ...


class DummyPoseEstimator:
    """Identity-rotation pose at the median depth of the bbox centre.

    This is *not* a real 6-DOF estimator. It exists so that downstream code
    can be wired and tested without pulling in GraspNet / FoundationPose
    weights, and so the protocol has a concrete reference. For a real
    deployment, replace with one of:

    * GraspNet-Baseline (point-cloud based, requires depth)
    * FoundationPose (RGB-D, requires CAD model)
    * MegaPose / OnePose (RGB-only with object templates)
    """

    def estimate(
        self,
        image_rgb: np.ndarray,
        depth: np.ndarray,
        bbox: np.ndarray,
    ) -> np.ndarray:
        bbox = np.asarray(bbox, dtype=np.float32).reshape(4)
        x1, y1, x2, y2 = bbox
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        # Median depth inside the bbox (robust to occlusion holes).
        h, w = depth.shape[:2]
        xi0 = int(max(0, np.floor(x1)))
        yi0 = int(max(0, np.floor(y1)))
        xi1 = int(min(w, np.ceil(x2)))
        yi1 = int(min(h, np.ceil(y2)))
        crop = depth[yi0:yi1, xi0:xi1]
        valid = crop[np.isfinite(crop) & (crop > 0)]
        z = float(np.median(valid)) if valid.size else 0.0

        pose = np.eye(4, dtype=np.float64)
        # Pixel-space "translation"; a real system would un-project via K^-1.
        pose[0, 3] = float(cx)
        pose[1, 3] = float(cy)
        pose[2, 3] = z
        return pose
