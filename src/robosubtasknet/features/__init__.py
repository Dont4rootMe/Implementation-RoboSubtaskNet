"""Feature extraction utilities for RoboSubtaskNet.

This subpackage exposes the I3D-R50 backbone wrapper used to extract the
1024-d RGB and flow features consumed by the attention-fusion + multi-stage
TCN. Optical-flow utilities live in ``flow`` (TV-L1, optional ``cv2``
dependency) and the CLI driver in ``extract``; neither is imported here so
that simply importing :mod:`robosubtasknet.features` does not require
``cv2`` or ``pytorchvideo`` to be installed.
"""

from .i3d import I3DFeatureExtractor

__all__ = ["I3DFeatureExtractor"]
