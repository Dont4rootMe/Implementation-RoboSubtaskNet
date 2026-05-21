"""RoboSubtaskNet — temporal sub-task segmentation for human-to-robot skill transfer.

A reimplementation of Sharma et al. (arXiv:2602.10015): a frozen I3D feature
backbone with learned attention fusion of RGB and optical-flow streams, refined
by a Fibonacci-dilated multi-stage temporal convolutional network and trained
under a composite cross-entropy + truncated MSE + transition-aware loss.

See ``IMPLEMENTATION_PLAN.md`` at the repository root for the full design spec.
"""

__version__ = "0.1.0"
