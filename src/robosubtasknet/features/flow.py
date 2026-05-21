"""Optical flow utilities for the I3D flow stream.

Two implementations are provided:

* :class:`TVL1Flow` — OpenCV's Dual TV-L1 algorithm. Matches the optical-flow
  distribution that the original Kinetics-pretrained I3D flow stream was
  trained on (clipped to +/- 20 pixels and rescaled to [-1, 1]). Slow on CPU
  (~1 fps for 224x224 frames) but distributionally faithful.

* :class:`RAFTFlow` — torchvision's ``raft_large`` model. High quality and
  GPU-accelerated, but outputs values on a different scale and statistics
  than TV-L1; feeding them into a Kinetics-pretrained I3D flow stream is an
  *out-of-distribution* operation and feature quality will degrade unless the
  flow head is fine-tuned. See Section 6.2 of IMPLEMENTATION_PLAN.md.

Both classes expose a callable interface returning ``np.ndarray`` of shape
``[H, W, 2]``. Helper :func:`compute_flow_from_frames` iterates over a stack
of frames and returns ``[T-1, H, W, 2]``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


__all__ = ["TVL1Flow", "RAFTFlow", "compute_flow_from_frames"]


class TVL1Flow:
    """Dual TV-L1 optical flow (Zach et al., 2007) via OpenCV's contrib module.

    Output values are clipped to ``+/- bound`` pixels and then rescaled to
    ``[-1, 1]`` to match the input normalization used when training the
    Kinetics-pretrained I3D flow stream (Carreira & Zisserman, 2017).

    Parameters
    ----------
    bound : int, default 20
        Maximum absolute pixel displacement to retain. Standard value from
        the I3D paper. After clipping, the flow is divided by ``bound`` so
        the output is in ``[-1, 1]``.

    Notes
    -----
    ``cv2`` is imported lazily inside :meth:`__init__` so that this module
    can be imported in environments without ``opencv-contrib-python``. A
    helpful :class:`ImportError` is raised if the contrib extras are missing.
    """

    def __init__(self, bound: int = 20) -> None:
        try:
            import cv2  # noqa: WPS433  (lazy import is intentional)
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "TVL1Flow requires opencv-contrib-python. Install with "
                "`pip install opencv-contrib-python` (the contrib build "
                "provides cv2.optflow)."
            ) from exc

        # The TV-L1 implementation lives in the `optflow` contrib module.
        if not hasattr(cv2, "optflow") or not hasattr(
            cv2.optflow, "DualTVL1OpticalFlow_create"
        ):
            raise ImportError(
                "TVL1Flow needs cv2.optflow.DualTVL1OpticalFlow_create, which "
                "is only shipped in opencv-contrib-python. Reinstall with "
                "`pip install --force-reinstall opencv-contrib-python` "
                "(and uninstall plain opencv-python if both are present)."
            )

        self._cv2 = cv2
        self.alg = cv2.optflow.DualTVL1OpticalFlow_create()
        self.bound = int(bound)

    def __call__(
        self, prev_gray: np.ndarray, curr_gray: np.ndarray
    ) -> np.ndarray:
        """Compute optical flow between two grayscale frames.

        Parameters
        ----------
        prev_gray, curr_gray : np.ndarray
            Single-channel ``uint8`` frames of identical shape ``[H, W]``.

        Returns
        -------
        np.ndarray
            Flow field of shape ``[H, W, 2]`` (dx, dy), dtype ``float32``,
            values clipped to ``+/- bound`` and rescaled to ``[-1, 1]``.
        """
        if prev_gray.ndim != 2 or curr_gray.ndim != 2:
            raise ValueError(
                "TVL1Flow expects grayscale frames of shape [H, W]; got "
                f"{prev_gray.shape} and {curr_gray.shape}."
            )
        if prev_gray.shape != curr_gray.shape:
            raise ValueError(
                "prev_gray and curr_gray must have matching shapes; got "
                f"{prev_gray.shape} vs {curr_gray.shape}."
            )

        flow = self.alg.calc(prev_gray, curr_gray, None)
        flow = np.clip(flow, -self.bound, self.bound) / float(self.bound)
        return flow.astype(np.float32)


class RAFTFlow:
    """RAFT optical flow (Teed & Deng, 2020) via ``torchvision``.

    Wraps :func:`torchvision.models.optical_flow.raft_large` with its
    matching transform. Outputs are returned as ``np.ndarray`` of shape
    ``[H, W, 2]`` for API parity with :class:`TVL1Flow`.

    .. warning::
       RAFT outputs are **not** drop-in compatible with the
       Kinetics-pretrained I3D flow stream. The original I3D flow weights
       expect TV-L1-style flow clipped to +/- 20 pixels and rescaled to
       ``[-1, 1]``. RAFT produces unbounded, higher-fidelity flow with
       different statistics, so feeding it directly to a frozen I3D flow
       head is an out-of-distribution operation and feature quality will
       degrade. Use RAFT only if you also (a) fine-tune the I3D flow head,
       (b) renormalize RAFT outputs to mimic TV-L1, or (c) accept the
       distribution mismatch. See Section 6.2 of IMPLEMENTATION_PLAN.md.

    Parameters
    ----------
    bound : int, default 20
        Clipping bound for output values, matching :class:`TVL1Flow`. RAFT
        flow is clipped to ``+/- bound`` and then divided by ``bound`` so
        the returned array is in ``[-1, 1]``. This makes the output range
        comparable to TV-L1 but does not fix the distributional mismatch.
    device : str, default "cuda"
        Torch device to run RAFT on. Falls back to CPU if CUDA is not
        available.
    iters : int, default 12
        Number of recurrent update iterations performed by RAFT.
    pretrained : bool, default True
        Whether to load the published RAFT-large weights.
    """

    def __init__(
        self,
        bound: int = 20,
        device: str = "cuda",
        iters: int = 12,
        pretrained: bool = True,
    ) -> None:
        try:
            import torch  # noqa: WPS433
            from torchvision.models.optical_flow import (
                Raft_Large_Weights,
                raft_large,
            )
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "RAFTFlow requires torch and torchvision (with "
                "optical-flow models). Install with "
                "`pip install torch torchvision`."
            ) from exc

        self._torch = torch
        self.bound = int(bound)
        self.iters = int(iters)

        # Resolve device, falling back gracefully when CUDA is missing.
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        weights = Raft_Large_Weights.DEFAULT if pretrained else None
        self.model = raft_large(weights=weights, progress=False).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # The torchvision-provided transform handles per-channel
        # normalization and ensures dtypes are correct.
        self._transform = (
            weights.transforms()
            if weights is not None
            else Raft_Large_Weights.DEFAULT.transforms()
        )

    def __call__(
        self, prev_rgb: np.ndarray, curr_rgb: np.ndarray
    ) -> np.ndarray:
        """Compute optical flow between two RGB frames.

        Parameters
        ----------
        prev_rgb, curr_rgb : np.ndarray
            ``uint8`` arrays of shape ``[H, W, 3]`` (channel-last) or
            ``[3, H, W]``. Must have identical shape.

        Returns
        -------
        np.ndarray
            Flow field of shape ``[H, W, 2]``, dtype ``float32``, clipped
            to ``+/- bound`` and rescaled to ``[-1, 1]``.
        """
        torch = self._torch

        prev_t = self._to_chw_tensor(prev_rgb)
        curr_t = self._to_chw_tensor(curr_rgb)
        if prev_t.shape != curr_t.shape:
            raise ValueError(
                "prev_rgb and curr_rgb must have matching shapes; got "
                f"{prev_t.shape} vs {curr_t.shape}."
            )

        # Add batch dim and run through torchvision's expected transform.
        prev_b = prev_t.unsqueeze(0).to(self.device)
        curr_b = curr_t.unsqueeze(0).to(self.device)
        prev_b, curr_b = self._transform(prev_b, curr_b)

        with torch.no_grad():
            preds = self.model(prev_b, curr_b, num_flow_updates=self.iters)
        # raft_large returns a list of flow predictions, one per update.
        flow_t = preds[-1] if isinstance(preds, (list, tuple)) else preds
        # flow_t: [1, 2, H, W] -> [H, W, 2]
        flow = flow_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        flow = np.clip(flow, -self.bound, self.bound) / float(self.bound)
        return flow.astype(np.float32)

    def _to_chw_tensor(self, frame: np.ndarray):
        torch = self._torch
        if frame.ndim != 3:
            raise ValueError(
                "RAFTFlow expects RGB frames of shape [H, W, 3] or "
                f"[3, H, W]; got {frame.shape}."
            )
        if frame.shape[-1] == 3 and frame.shape[0] != 3:
            arr = np.ascontiguousarray(frame.transpose(2, 0, 1))
        elif frame.shape[0] == 3:
            arr = np.ascontiguousarray(frame)
        else:
            raise ValueError(
                "RAFTFlow expects RGB frames with 3 channels; got shape "
                f"{frame.shape}."
            )
        tensor = torch.from_numpy(arr)
        if tensor.dtype != torch.uint8:
            tensor = tensor.to(torch.uint8)
        return tensor


def compute_flow_from_frames(
    frames: np.ndarray, method: Literal["tvl1", "raft"] = "tvl1"
) -> np.ndarray:
    """Compute consecutive-frame optical flow for a clip.

    Parameters
    ----------
    frames : np.ndarray
        Stack of frames with shape ``[T, H, W, 3]`` (RGB, ``uint8``).
    method : {"tvl1", "raft"}, default "tvl1"
        Which estimator to use. See :class:`TVL1Flow` / :class:`RAFTFlow`
        for caveats — particularly the distributional mismatch between
        RAFT and the Kinetics-pretrained I3D flow stream.

    Returns
    -------
    np.ndarray
        Flow tensor of shape ``[T - 1, H, W, 2]``, dtype ``float32``,
        values in ``[-1, 1]``.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            "frames must have shape [T, H, W, 3] with C=3 (RGB); got "
            f"{frames.shape}."
        )
    T = frames.shape[0]
    if T < 2:
        raise ValueError(
            f"Need at least 2 frames to compute optical flow; got T={T}."
        )

    method_lc = method.lower()
    if method_lc == "tvl1":
        try:
            import cv2  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "TV-L1 flow requires opencv-contrib-python. Install with "
                "`pip install opencv-contrib-python`."
            ) from exc

        estimator = TVL1Flow()
        # Pre-convert to grayscale once to avoid recomputation.
        grays = [
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames
        ]
        out = np.empty((T - 1, frames.shape[1], frames.shape[2], 2), dtype=np.float32)
        for t in range(T - 1):
            out[t] = estimator(grays[t], grays[t + 1])
        return out

    if method_lc == "raft":
        estimator = RAFTFlow()
        out = np.empty((T - 1, frames.shape[1], frames.shape[2], 2), dtype=np.float32)
        for t in range(T - 1):
            out[t] = estimator(frames[t], frames[t + 1])
        return out

    raise ValueError(
        f"Unknown optical-flow method {method!r}; expected 'tvl1' or 'raft'."
    )
