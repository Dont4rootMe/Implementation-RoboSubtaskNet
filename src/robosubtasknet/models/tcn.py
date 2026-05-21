"""Fibonacci-dilated single-stage TCN building blocks.

Reference: IMPLEMENTATION_PLAN.md §8 (RoboSubtaskNet, arXiv:2602.10015).

This module implements the per-stage TCN used by ``RoboSubtaskNet``:

* :func:`fibonacci` — produces the dilation schedule
  ``[F_2, F_3, ..., F_{n+1}]`` skipping the duplicated leading ``F_1=1``.
* :class:`FibonacciDilatedLayer` — a single residual dilated conv block
  (dilated 3x3 conv -> ReLU -> 1x1 conv -> dropout -> add residual).
* :class:`SingleStageTCN` — stacks ``num_layers`` of the above with a 1x1
  input projection and a 1x1 classification head.
* :func:`receptive_field` — closed-form receptive field for ``num_layers``
  Fibonacci-dilated kernel-3 convolutions.

Multi-stage refinement (Section 8.3) is composed in ``robosubtasknet.py``
by another agent and is intentionally not implemented here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def fibonacci(n: int) -> list[int]:
    """Return the Fibonacci dilation schedule ``[F_2, F_3, ..., F_{n+1}]``.

    The sequence skips the duplicated leading ``F_1 = 1`` so that the
    dilation pattern grows monotonically. For example:

    >>> fibonacci(10)
    [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    >>> fibonacci(1)
    [1]
    >>> fibonacci(2)
    [1, 2]
    >>> fibonacci(0)
    []

    Args:
        n: Number of dilation values to produce (i.e. the number of
            ``FibonacciDilatedLayer`` instances in a stage). Must be
            non-negative.

    Returns:
        A list of ``n`` integers ``[F_2, F_3, ..., F_{n+1}]``.
    """
    fib = [1, 1]
    while len(fib) < n + 2:
        fib.append(fib[-1] + fib[-2])
    return fib[1:n + 1]  # F_2 .. F_{n+1}


def receptive_field(num_layers: int, kernel_size: int = 3) -> int:
    """Compute the temporal receptive field of a Fibonacci-dilated stage.

    For a stack of ``num_layers`` dilated convolutions with kernel size
    ``k`` and dilations ``d_l = F_{l+1}``, the receptive field at the
    output of the stage is::

        RF(L) = 1 + (k - 1) * sum_{l=1..L} F_{l+1}

    The plan (Section 8.1) specializes this to ``k = 3``:

        RF(L) = 1 + 2 * sum_{l=1..L} F_{l+1}

    For ``L = 10`` this returns ``463`` frames at I3D's output stride.

    Args:
        num_layers: Number of dilated layers ``L`` in the stage.
        kernel_size: Convolution kernel size ``k``. Defaults to 3 (the
            value used throughout RoboSubtaskNet).

    Returns:
        The integer receptive field at the stage's output.
    """
    return 1 + (kernel_size - 1) * sum(fibonacci(num_layers))


class FibonacciDilatedLayer(nn.Module):
    """Residual dilated 1D conv block with a Fibonacci-scheduled dilation.

    Structure (Section 8.2):

    ``x -> Conv1d(k=3, dilation=d, padding=d) -> ReLU
         -> Conv1d(k=1) -> Dropout -> + x``

    The ``padding = dilation`` choice preserves the temporal length
    (this is symmetric, "same"-style padding, not strictly causal — it
    matches the MS-TCN convention the plan inherits).

    Args:
        dim: Channel dimension of the residual stream (input == output).
        dilation: Dilation factor for the 3x3 convolution.
        dropout: Dropout probability applied after the 1x1 projection.
    """

    def __init__(self, dim: int, dilation: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            dim, dim, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual dilated block.

        Args:
            x: Input tensor of shape ``[B, D, T]``.

        Returns:
            Output tensor of shape ``[B, D, T]`` (same length as ``x``).
        """
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class SingleStageTCN(nn.Module):
    """A single stage of the Fibonacci-dilated multi-stage TCN.

    Composition (Section 8.2):

    ``input -> Conv1d(in_dim -> hidden_dim, k=1)
            -> [FibonacciDilatedLayer(hidden_dim, d) for d in fibonacci(L)]
            -> Conv1d(hidden_dim -> num_classes, k=1)``

    Args:
        num_layers: Number of dilated layers ``L`` in the stage. Drives
            the dilation schedule via :func:`fibonacci`.
        in_dim: Input channel dimension (``D_in`` in the docstrings).
        hidden_dim: Channel dimension of the residual stream (MS-TCN
            convention is 64).
        num_classes: Number of output classes ``C``.
        dropout: Dropout probability inside each
            :class:`FibonacciDilatedLayer`.
    """

    def __init__(
        self,
        num_layers: int,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        dilations = fibonacci(num_layers)
        self.proj_in = nn.Conv1d(in_dim, hidden_dim, kernel_size=1)
        self.layers = nn.ModuleList(
            FibonacciDilatedLayer(hidden_dim, d, dropout) for d in dilations
        )
        self.head = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a single-stage forward pass.

        Args:
            x: Input tensor of shape ``[B, D_in, T]``.

        Returns:
            Logits of shape ``[B, C, T]`` (no softmax applied).
        """
        h = self.proj_in(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)
