"""Phase 2 robot-execution scaffolds for RoboSubtaskNet.

These modules wire predicted sub-task labels to actual robot motion:

* :mod:`.dmp` — Dynamic Movement Primitives (one per sub-task) for replaying
  demonstrated motions with goal re-parameterisation.
* :mod:`.detection` — Object-detection / 6-DOF pose-estimation interfaces
  used to ground object-centric primitives (``reach``, ``pick`` …).
* :mod:`.servoing` — Image-based visual servoing for closed-loop refinement
  on ``reach`` and ``wipe``.

This sub-package is independent of the segmentation pipeline and may be
absent in deployments that only need offline labelling. See Section 12 of
``IMPLEMENTATION_PLAN.md`` for the full design.
"""

from .detection import (
    Detection,
    DummyPoseEstimator,
    ObjectDetector,
    PoseEstimator6D,
    YOLOv8Detector,
)
from .dmp import DiscreteDMP, MultiDOFDMP, SubtaskDMPLibrary
from .servoing import IBVSController, build_image_jacobian

__all__ = [
    # DMP
    "DiscreteDMP",
    "MultiDOFDMP",
    "SubtaskDMPLibrary",
    # Detection / pose
    "Detection",
    "ObjectDetector",
    "YOLOv8Detector",
    "PoseEstimator6D",
    "DummyPoseEstimator",
    # Servoing
    "IBVSController",
    "build_image_jacobian",
]
