"""Augmentations for RoboSubtaskNet.

Two categories of transforms:

1. ``Feature-level``: operate on per-frame RGB / flow feature tensors
   (``[T, D]`` with ``D == 1024`` for I3D-R50). Composed by ``data/dataset``
   during training.
2. ``Video-frame``: operate on raw frames; used inside the feature-extraction
   pipeline when features are computed from videos rather than loaded.

All feature-level transforms share the signature
``(rgb, flow, labels, mask) -> (rgb, flow, labels, mask)`` where ``rgb`` /
``flow`` are ``[T, D]`` floats, ``labels`` is ``[T]`` int64, and ``mask`` is
``[T]`` bool/float (1 = valid). Returning the mask makes it easy to chain
transforms that change ``T``.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision.transforms import ColorJitter

FeatureSample = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


# --------------------------------------------------------------------------- #
# Feature-level augmentations
# --------------------------------------------------------------------------- #
class FeatureDropout:
    """Stochastic channel- and temporal-frame dropout on feature streams.

    Two independent Bernoulli masks are sampled:

    * ``p_channel``: probability of zeroing each of the ``D`` feature
      channels for the entire clip (applied identically to all frames).
    * ``p_time``: probability of zeroing the *entire* feature vector at a
      given frame (i.e. dropping that timestep's features). The frame is not
      removed from the sequence; only its features are zeroed. The mask is
      left untouched so that downstream losses still receive a label there.

    Both RGB and flow streams use the same temporal-frame mask so that
    fusion stays aligned, but receive independent channel masks.
    """

    def __init__(self, p_channel: float = 0.0, p_time: float = 0.0) -> None:
        if not 0.0 <= p_channel < 1.0:
            raise ValueError(f"p_channel must be in [0, 1); got {p_channel}")
        if not 0.0 <= p_time < 1.0:
            raise ValueError(f"p_time must be in [0, 1); got {p_time}")
        self.p_channel = float(p_channel)
        self.p_time = float(p_time)

    def __call__(
        self,
        rgb: torch.Tensor,
        flow: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> FeatureSample:
        T, D = rgb.shape
        if self.p_channel > 0.0:
            ch_rgb = (torch.rand(D, device=rgb.device) >= self.p_channel).to(rgb.dtype)
            ch_flow = (torch.rand(D, device=flow.device) >= self.p_channel).to(flow.dtype)
            rgb = rgb * ch_rgb
            flow = flow * ch_flow
        if self.p_time > 0.0:
            t_keep = (torch.rand(T, device=rgb.device) >= self.p_time).to(rgb.dtype)
            rgb = rgb * t_keep.unsqueeze(-1)
            flow = flow * t_keep.unsqueeze(-1).to(flow.dtype)
        return rgb, flow, labels, mask


class GaussianFeatureNoise:
    """Add zero-mean Gaussian noise to RGB and flow features independently."""

    def __init__(self, sigma: float = 0.01) -> None:
        if sigma < 0.0:
            raise ValueError(f"sigma must be non-negative; got {sigma}")
        self.sigma = float(sigma)

    def __call__(
        self,
        rgb: torch.Tensor,
        flow: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> FeatureSample:
        if self.sigma == 0.0:
            return rgb, flow, labels, mask
        rgb = rgb + torch.randn_like(rgb) * self.sigma
        flow = flow + torch.randn_like(flow) * self.sigma
        return rgb, flow, labels, mask


class TemporalSubsample:
    """Subsample frames along the temporal axis with a random offset.

    Useful for shortening long clips during training. ``stride`` controls the
    base downsampling factor; ``jitter`` perturbs the starting offset in
    ``[0, jitter]`` (inclusive) so the sampled indices vary across epochs.
    Labels and mask are subsampled identically to keep alignment.
    """

    def __init__(self, stride: int = 1, jitter: int = 0) -> None:
        if stride < 1:
            raise ValueError(f"stride must be >= 1; got {stride}")
        if jitter < 0:
            raise ValueError(f"jitter must be non-negative; got {jitter}")
        self.stride = int(stride)
        self.jitter = int(jitter)

    def __call__(
        self,
        rgb: torch.Tensor,
        flow: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> FeatureSample:
        T = rgb.shape[0]
        if self.stride == 1 and self.jitter == 0:
            return rgb, flow, labels, mask
        max_off = min(self.jitter, max(T - 1, 0))
        offset = int(torch.randint(0, max_off + 1, (1,)).item()) if max_off > 0 else 0
        idx = torch.arange(offset, T, self.stride, device=rgb.device)
        return rgb[idx], flow[idx], labels[idx], mask[idx]


class Compose:
    """Chain feature-level augmentations sequentially."""

    def __init__(self, transforms: List[object]) -> None:
        self.transforms = list(transforms)

    def __call__(
        self,
        rgb: torch.Tensor,
        flow: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> FeatureSample:
        for t in self.transforms:
            rgb, flow, labels, mask = t(rgb, flow, labels, mask)
        return rgb, flow, labels, mask


# --------------------------------------------------------------------------- #
# Video-frame augmentations (used during feature extraction)
# --------------------------------------------------------------------------- #
class RandomColorJitter(nn.Module):
    """Per-frame photometric jitter (brightness, contrast, saturation).

    Thin wrapper around ``torchvision.transforms.ColorJitter`` so it can be
    plugged into the feature-extraction pipeline. Operates on uint8 or float
    image tensors of shape ``[C, H, W]`` or ``[..., C, H, W]``.
    """

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
    ) -> None:
        super().__init__()
        self.jitter = ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation
        )

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        return self.jitter(frame)


class RandomHorizontalFlip(nn.Module):
    """Random horizontal flip for video frames.

    .. warning::
        Horizontal flipping is **only** safe when the paired flow stream's
        x-component is sign-flipped to match the geometric flip. Failing to
        do so injects a systematic disagreement between RGB and flow streams
        and will degrade I3D feature quality. This module flips the RGB
        frame only; if a flow tensor is supplied via ``flow``, its
        x-component (channel 0) is sign-flipped as well. Otherwise the
        caller is responsible for handling the flow side.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1]; got {p}")
        self.p = float(p)

    def forward(
        self,
        frame: torch.Tensor,
        flow: torch.Tensor | None = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if torch.rand(1).item() < self.p:
            frame = torch.flip(frame, dims=[-1])
            if flow is not None:
                flow = torch.flip(flow, dims=[-1])
                # Flip x-component sign (channel 0 of the 2-channel flow).
                flow = flow.clone()
                flow[..., 0, :, :] = -flow[..., 0, :, :]
        if flow is not None:
            return frame, flow
        return frame
