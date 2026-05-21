"""I3D-R50 feature extractor wrapper.

Loads the Kinetics-400 pretrained I3D-R50 from ``pytorchvideo`` and exposes a
small, frozen feature extractor used throughout the RoboSubtaskNet pipeline.

The Kinetics-pretrained I3D backbone has an **8x temporal downsampling** factor:
for an input clip of ``T`` frames the network produces ``ceil(T / 8)`` feature
vectors of dimension 1024. The standard MS-TCN / RoboSubtaskNet recipe uses a
sliding 16-frame window with stride 8 over the raw video, which yields one
1024-d feature per 8 input frames (i.e. ~1 feature every ~0.27 s at 30 fps).

References
----------
- Carreira & Zisserman, "Quo Vadis, Action Recognition? A New Model and the
  Kinetics Dataset", CVPR 2017.
- pytorchvideo ``i3d_r50`` model card.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# Kinetics-400 RGB statistics used by the pretrained I3D weights. These are
# the same values used by pytorchvideo's stock transforms for I3D-R50.
_KINETICS_RGB_MEAN = (0.45, 0.45, 0.45)
_KINETICS_RGB_STD = (0.225, 0.225, 0.225)

# I3D temporal stride: the network downsamples T by a factor of 8 in time
# (one stride-2 op in the stem plus two stride-2 ops over the res-blocks).
I3D_TEMPORAL_STRIDE: int = 8

# I3D-R50 final feature dimension (pool5 output channel count).
I3D_FEATURE_DIM: int = 1024


class I3DFeatureExtractor(nn.Module):
    """Extract 1024-d features per temporal window from a frozen I3D-R50.

    Parameters
    ----------
    modality:
        ``"rgb"`` (3 input channels) or ``"flow"`` (2 input channels). The
        Kinetics-pretrained I3D ships separate RGB and flow streams; this
        wrapper currently exposes the RGB weights from ``pytorchvideo`` for
        both modalities. For the flow stream, callers must arrange weights
        externally (see the flow caveat in :meth:`normalize_flow`).
    pretrained:
        If True, load Kinetics-400 pretrained weights via pytorchvideo's hub.
    freeze:
        If True (default), call ``eval()`` and set ``requires_grad_(False)``
        on every parameter. Set to False if you intend to fine-tune.

    Input
    -----
    Tensor of shape ``[B, C, T, H, W]`` with ``C=3`` for RGB or ``C=2`` for
    flow. Spatial size should be 224x224 to match Kinetics pretraining.

    Output
    ------
    Tensor of shape ``[B, T_out, 1024]`` where ``T_out = ceil(T / 8)``.

    Notes
    -----
    pytorchvideo must be installed at construction time. The import is lazy
    so this module can be imported without ``pytorchvideo`` present (e.g. for
    unit tests of unrelated components). Constructing the class without
    ``pytorchvideo`` raises an informative :class:`ImportError`.
    """

    def __init__(
        self,
        modality: Literal["rgb", "flow"] = "rgb",
        pretrained: bool = True,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        if modality not in ("rgb", "flow"):
            raise ValueError(
                f"modality must be 'rgb' or 'flow', got {modality!r}"
            )

        # Lazy import so that ``import robosubtasknet.features.i3d`` works in
        # environments without pytorchvideo (only constructing the class is
        # gated by the dependency).
        try:
            from pytorchvideo.models.hub import i3d_r50  # type: ignore
        except ImportError as e:  # pragma: no cover - exercised at runtime
            raise ImportError(
                "I3DFeatureExtractor requires the 'pytorchvideo' package. "
                "Install it with `pip install pytorchvideo` (and a matching "
                "PyTorch version)."
            ) from e

        self.modality = modality
        self.pretrained = pretrained
        self.freeze = freeze
        self.feature_dim = I3D_FEATURE_DIM
        self.temporal_stride = I3D_TEMPORAL_STRIDE

        self.model = i3d_r50(pretrained=pretrained)

        # Strip the classification head: replace the 400-way linear projector
        # and its softmax/activation with identities so that ``forward``
        # returns the 1024-d pool5 features.
        last_block = self.model.blocks[-1]
        if hasattr(last_block, "proj") and last_block.proj is not None:
            last_block.proj = nn.Identity()
        if hasattr(last_block, "activation") and last_block.activation is not None:
            last_block.activation = nn.Identity()
        # pytorchvideo's ResNetBasicHead may apply an output pooling that
        # already collapses (T, H, W) to a single feature; keep its default
        # behavior so we get one feature per temporal window position.

        if freeze:
            self.model.eval()
            for p in self.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a clip through I3D-R50 and return pool5 features.

        Parameters
        ----------
        x:
            Tensor of shape ``[B, C, T, H, W]``. ``C`` must equal 3 for
            ``modality="rgb"`` and 2 for ``modality="flow"``.

        Returns
        -------
        Tensor of shape ``[B, T_out, 1024]`` where ``T_out = ceil(T / 8)``
        due to I3D's 8x temporal downsampling.
        """
        if x.dim() != 5:
            raise ValueError(
                f"Expected 5D input [B, C, T, H, W], got shape {tuple(x.shape)}"
            )
        expected_c = 3 if self.modality == "rgb" else 2
        if x.shape[1] != expected_c:
            raise ValueError(
                f"Expected C={expected_c} channels for modality "
                f"{self.modality!r}, got C={x.shape[1]}"
            )

        feat = self.model(x)

        # pytorchvideo's classification head usually outputs either
        # ``[B, num_classes]`` (after pooling) or ``[B, C, T_out, 1, 1]``
        # before pooling. With the head replaced by identities we get the
        # pre-projection feature map; normalize to ``[B, T_out, 1024]``.
        if feat.dim() == 5:
            # [B, C, T_out, H_out, W_out] -> spatially pool and transpose.
            feat = feat.mean(dim=(-2, -1))  # [B, C, T_out]
            feat = feat.transpose(1, 2).contiguous()  # [B, T_out, C]
        elif feat.dim() == 3:
            # [B, C, T_out] -> [B, T_out, C]
            feat = feat.transpose(1, 2).contiguous()
        elif feat.dim() == 2:
            # [B, C] -> [B, 1, C] (whole-clip feature; happens when the head
            # already applied global average pooling).
            feat = feat.unsqueeze(1)
        else:
            raise RuntimeError(
                f"Unexpected I3D output rank {feat.dim()} with shape "
                f"{tuple(feat.shape)}"
            )

        return feat

    @torch.no_grad()
    def extract_clip_features(
        self,
        video_chw_t: torch.Tensor,
        window: int = 16,
        stride: int = 8,
        batch_size: int = 8,
    ) -> torch.Tensor:
        """Sliding-window feature extraction over a (possibly long) clip.

        Slides a ``window``-frame window with the given ``stride`` over the
        temporal axis and aggregates per-window features into a single
        ``[T_feat, 1024]`` tensor. When the final window would extend past
        the end of the clip the last frame is repeated (edge-replication
        padding) so the trailing portion of the video still contributes a
        feature vector.

        Parameters
        ----------
        video_chw_t:
            Tensor of shape ``[C, T, H, W]`` (no batch dim). ``C`` must
            equal 3 for RGB or 2 for flow, matching ``self.modality``. The
            spatial size should be 224x224 and the tensor should already be
            normalized (see :meth:`normalize_rgb` / :meth:`normalize_flow`).
        window:
            Number of frames per I3D input (default 16, matching Kinetics
            pretraining).
        stride:
            Hop in frames between consecutive windows (default 8, matching
            the I3D 8x temporal stride so the global feature rate equals one
            feature per ``stride`` input frames).
        batch_size:
            Number of windows to forward through the network per pass.
            Larger values are faster on GPU but use more memory.

        Returns
        -------
        Tensor of shape ``[T_feat, 1024]`` on the same device / dtype as
        ``video_chw_t``. ``T_feat`` is ``ceil(max(T - window, 0) / stride) + 1``
        when ``T >= window``; for shorter clips a single feature is returned
        after padding.
        """
        if video_chw_t.dim() != 4:
            raise ValueError(
                f"Expected 4D input [C, T, H, W], got shape "
                f"{tuple(video_chw_t.shape)}"
            )
        if window <= 0 or stride <= 0:
            raise ValueError(
                f"window and stride must be positive (got window={window}, "
                f"stride={stride})"
            )

        c, t, h, w = video_chw_t.shape
        expected_c = 3 if self.modality == "rgb" else 2
        if c != expected_c:
            raise ValueError(
                f"Expected C={expected_c} channels for modality "
                f"{self.modality!r}, got C={c}"
            )

        # Edge-pad along T so we have at least ``window`` frames and the
        # number of frames is congruent for the final window.
        if t < window:
            pad = window - t
            video_chw_t = F.pad(video_chw_t, (0, 0, 0, 0, 0, pad), mode="replicate")
            t = window
        else:
            # If (t - window) is not a multiple of stride, pad the tail so
            # that the last window aligns with the end of the clip.
            remainder = (t - window) % stride
            if remainder != 0:
                pad = stride - remainder
                video_chw_t = F.pad(
                    video_chw_t, (0, 0, 0, 0, 0, pad), mode="replicate"
                )
                t = t + pad

        # Build the list of window start indices.
        starts = list(range(0, t - window + 1, stride))
        if not starts:
            starts = [0]

        device = video_chw_t.device

        outputs: list[torch.Tensor] = []
        for batch_start in range(0, len(starts), batch_size):
            batch_starts = starts[batch_start : batch_start + batch_size]
            clips = torch.stack(
                [video_chw_t[:, s : s + window] for s in batch_starts], dim=0
            )  # [B, C, window, H, W]
            feats = self.forward(clips.to(device))  # [B, T_out, 1024]
            # I3D collapses the 16-frame window into 2 features at stride 8;
            # average across those to get a single feature per window
            # position so the output rate equals ``len(starts)`` features.
            if feats.shape[1] > 1:
                feats = feats.mean(dim=1, keepdim=True)
            feats = feats.squeeze(1)  # [B, 1024]
            outputs.append(feats)

        return torch.cat(outputs, dim=0)  # [T_feat, 1024]

    # ------------------------------------------------------------------ #
    # Preprocessing helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def normalize_rgb(x: torch.Tensor) -> torch.Tensor:
        """Apply Kinetics RGB normalization.

        Expects ``x`` with values in ``[0, 1]`` and shape ``[..., 3, ...]``
        with the channel dim somewhere; the canonical layout used in this
        codebase is ``[B, 3, T, H, W]`` (or ``[3, T, H, W]`` without a
        batch). The function broadcasts over leading and trailing dims.

        Mean / std follow pytorchvideo's stock I3D-R50 transform:
        ``mean = (0.45, 0.45, 0.45)``, ``std = (0.225, 0.225, 0.225)``.
        """
        if x.dim() < 2:
            raise ValueError(
                f"normalize_rgb expects at least 2 dims, got shape "
                f"{tuple(x.shape)}"
            )
        # Locate the channel axis: assume the first axis of size 3 from the
        # left is the channel dim. In practice this is always dim 0 or 1.
        channel_axis = None
        for axis, size in enumerate(x.shape):
            if size == 3:
                channel_axis = axis
                break
        if channel_axis is None:
            raise ValueError(
                f"normalize_rgb could not find a channel dim of size 3 in "
                f"shape {tuple(x.shape)}"
            )

        mean = torch.as_tensor(_KINETICS_RGB_MEAN, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(_KINETICS_RGB_STD, dtype=x.dtype, device=x.device)
        view = [1] * x.dim()
        view[channel_axis] = 3
        mean = mean.view(view)
        std = std.view(view)
        return (x - mean) / std

    @staticmethod
    def normalize_flow(x: torch.Tensor, bound: float = 20.0) -> torch.Tensor:
        """Apply Kinetics-I3D flow normalization.

        Clamps optical-flow displacements to ``[-bound, bound]`` pixels (the
        original Kinetics-I3D recipe uses ``bound=20``) and divides by
        ``bound`` so the values land in ``[-1, 1]``.

        Important caveat
        ----------------
        The Kinetics-pretrained I3D flow stream was trained on **TV-L1**
        optical flow computed with a specific OpenCV configuration. Feeding
        flow from a different estimator (e.g. RAFT) without renormalization
        leads to distributional drift and degraded features. For RAFT, you
        should either re-fine-tune the flow head or match TV-L1 statistics
        empirically before relying on these features. See §6.2 of the
        implementation plan.
        """
        return torch.clamp(x, -bound, bound) / bound


__all__ = [
    "I3DFeatureExtractor",
    "I3D_FEATURE_DIM",
    "I3D_TEMPORAL_STRIDE",
]
