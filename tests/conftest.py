"""Shared pytest fixtures for the RoboSubtaskNet test suite.

The fixtures here keep individual test files compact: each test that needs
a synthetic batch of two-stream features, a tiny label vocabulary, or a
small "allowed transition" matrix can pull a ready-made object from this
module instead of re-creating one. Sizes are deliberately small so the
CPU-only CI image runs the full unit-test sweep in a couple of seconds.

Markers registered here (also declared in ``pyproject.toml`` under
``[tool.pytest.ini_options]``):

* ``gpu`` — for tests that require a CUDA device. The collection hook
  below auto-skips these when ``torch.cuda.is_available()`` is False so
  the same suite is safe to run on a laptop or in a CPU-only CI runner.

Fixture parameter choices:

* ``B = 2`` — batch size large enough to exercise per-sample broadcasting.
* ``T = 32`` — short enough to keep tensors tiny but ``> 1`` so that the
  T-MSE / transition losses (which compare consecutive frames) have
  multiple pairs to average over.
* ``D = 64`` — matches the MS-TCN "hidden" convention and is enough to
  catch shape bugs without burning RAM.
* ``num_classes = 5`` — small but ``> 2`` so the grammar matrix is not a
  trivial 2x2 case.
* Random tensors use a per-fixture-call ``torch.Generator`` seeded with
  ``0`` for deterministic test failures.
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Marker registration & GPU autoskip
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``gpu`` marker.

    ``pyproject.toml`` already declares the marker under
    ``[tool.pytest.ini_options].markers`` (so ``--strict-markers`` is
    happy) but re-registering here keeps the test suite self-describing
    when the file is run with a bare ``pytest`` outside the project.
    """
    config.addinivalue_line(
        "markers",
        "gpu: marks tests that require a CUDA device (auto-skipped if no GPU).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-skip ``@pytest.mark.gpu`` tests on machines without CUDA."""
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA device not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


# ---------------------------------------------------------------------------
# Static fixtures (no randomness, no I/O)
# ---------------------------------------------------------------------------


@pytest.fixture
def device() -> torch.device:
    """Default device used by unit tests — always CPU.

    Tests that exercise the GPU path opt in explicitly via
    ``@pytest.mark.gpu`` and build their own ``cuda`` device.
    """
    return torch.device("cpu")


@pytest.fixture
def num_classes() -> int:
    """Tiny class vocabulary used across loss / metric tests (``C = 5``)."""
    return 5


# ---------------------------------------------------------------------------
# Synthetic batch fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_batch(
    device: torch.device, num_classes: int
) -> dict[str, torch.Tensor]:
    """Synthetic two-stream batch matching ``RoboSubtaskNet``'s input contract.

    Shapes (kept deliberately small for CPU CI):

    * ``rgb``    — ``[B, T, D]`` float32, standard-normal random values.
    * ``flow``   — ``[B, T, D]`` float32, standard-normal random values.
    * ``labels`` — ``[B, T]`` int64, uniform random in ``[0, num_classes)``.
    * ``mask``   — ``[B, T]`` float32, all ones (no padding by default).

    Both streams are drawn from the same deterministic generator so that
    test failures reproduce verbatim. Tests that need *identical* RGB and
    flow streams (e.g. the fusion invariant from §7.3) should derive their
    own pair from ``rgb`` rather than rely on this fixture.
    """
    g = torch.Generator(device=device.type).manual_seed(0)
    B, T, D = 2, 32, 64
    rgb = torch.randn(B, T, D, generator=g, device=device)
    flow = torch.randn(B, T, D, generator=g, device=device)
    labels = torch.randint(
        low=0, high=num_classes, size=(B, T), generator=g, device=device,
        dtype=torch.int64,
    )
    mask = torch.ones(B, T, dtype=torch.float32, device=device)
    return {"rgb": rgb, "flow": flow, "labels": labels, "mask": mask}


@pytest.fixture
def tiny_logits(
    device: torch.device, num_classes: int
) -> torch.Tensor:
    """Per-frame logits tensor of shape ``[B, C, T]``.

    Matches the channels-first layout that ``SingleStageTCN`` /
    ``RoboSubtaskNet`` emit and that the loss modules consume. Sampled
    from a standard normal with a fixed seed for reproducibility.
    """
    g = torch.Generator(device=device.type).manual_seed(1)
    B, T = 2, 32
    return torch.randn(B, num_classes, T, generator=g, device=device)


@pytest.fixture
def tiny_allowed_transitions(num_classes: int) -> torch.Tensor:
    """Tiny ``[C, C]`` bool transition matrix for the transition loss.

    Construction:

    * Identity — every self-transition ``(c, c)`` is allowed (required by
      ``TransitionLoss``: without this, every persistent sub-task would
      be penalized).
    * Two extra ordered transitions are turned on: ``(0, 1)`` and
      ``(1, 2)``. This leaves several forbidden off-diagonal pairs (e.g.
      ``(0, 2)``, ``(2, 0)``, ``(3, 4)``, ...) which the tests use to
      assert that flipping between forbidden labels lifts the loss above
      zero.
    """
    allowed = torch.eye(num_classes, dtype=torch.bool)
    # A couple of explicit allowed transitions, picked so that several
    # other pairs remain forbidden for the positivity test.
    allowed[0, 1] = True
    allowed[1, 2] = True
    return allowed
