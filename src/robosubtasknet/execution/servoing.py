"""Image-based visual servoing (IBVS) for closed-loop reach / wipe primitives.

Section 12.3 of the implementation plan specifies a proportional controller
of the form

.. math::
    \\dot q = J^+_{\\text{img}}(s) \\cdot K_p \\cdot (s^{*} - s)

where :math:`s, s^{*} \\in \\mathbb R^{2N}` stack ``N`` point-features as
``(u_i, v_i)`` in normalised image coordinates, :math:`J_{\\text{img}}` is the
2N × 6 image Jacobian (a.k.a. interaction matrix), and :math:`K_p` is a
diagonal gain. Solving for the end-effector twist gives

.. math::
    v = K_p \\cdot J^+ \\cdot (s^{*} - s)

with :math:`v \\in \\mathbb R^6` the body-frame twist
``[v_x, v_y, v_z, ω_x, ω_y, ω_z]^T`` to send through the robot's velocity
controller. The Jacobian for one point feature at normalised coordinates
``(x, y)`` and depth ``Z`` (Chaumette & Hutchinson, 2006) is

.. math::
    L_p(x, y, Z) = \\begin{bmatrix}
        -1/Z &  0      &  x/Z   &  xy        & -(1 + x^2) & y \\\\
         0   & -1/Z    &  y/Z   &  1 + y^2   & -xy        & -x
    \\end{bmatrix}

and we stack ``N`` such blocks vertically to form ``L_e``.

Real-time caveats (see Section 12.4 of the implementation plan)
---------------------------------------------------------------

* Segmentation inference for RoboSubtaskNet is fast (<50 ms / s of video on
  a single GPU); the slow part of the visual pipeline is feature
  extraction (I3D + TV-L1), which does *not* run real-time on HD video.
* For real-time deployment, use a lighter video encoder (X3D-S or MoViNet)
  or process at lower resolution / frame rate. The paper-faithful stack
  achieves only ~1 fps of sub-task updates, which is adequate for slow
  manipulation but not for fast servoing.
* The IBVS controller in this module assumes the underlying robot driver
  accepts a Cartesian twist command at a steady rate (e.g. 100 Hz). The
  *sub-task* changes only at ~1 Hz; the servoing loop must therefore run
  at its own clock and only re-fetch the goal feature set when a new
  sub-task / new detection arrives.
* The depth used to build the Jacobian must be reasonably accurate —
  IBVS is robust to small depth errors but degrades when ``Z`` is mis-
  estimated by more than ~30%. Prefer measured depth from the RGB-D
  sensor over a cached / constant value when available.

References
----------
Chaumette, F. & Hutchinson, S. *Visual servo control. I. Basic
approaches.* IEEE Robotics & Automation Magazine, 13(4):82–90, 2006.
"""

from __future__ import annotations

import numpy as np

__all__ = ["IBVSController", "build_image_jacobian"]


def build_image_jacobian(
    features: np.ndarray,
    depth: float | np.ndarray,
) -> np.ndarray:
    """Construct the stacked image Jacobian ``L_e`` for ``N`` point features.

    Parameters
    ----------
    features
        Flat ``(2N,)`` or stacked ``(N, 2)`` array of ``(x, y)`` features in
        normalised image coordinates (i.e. already pre-multiplied by ``K^-1``;
        if you only have pixel coordinates, do the un-projection upstream).
    depth
        Either a scalar depth applied to every feature, or a ``(N,)`` per-
        feature depth array. Positive values, in metres.

    Returns
    -------
    np.ndarray
        Jacobian of shape ``(2N, 6)``.
    """
    pts = np.asarray(features, dtype=np.float64).reshape(-1, 2)
    n = pts.shape[0]
    if n == 0:
        return np.zeros((0, 6), dtype=np.float64)

    if np.isscalar(depth):
        z = np.full(n, float(depth))
    else:
        z = np.asarray(depth, dtype=np.float64).reshape(-1)
        if z.size != n:
            raise ValueError(
                f"depth has {z.size} entries but {n} features were provided"
            )
    # Numerical floor on depth to keep 1/Z finite.
    z = np.where(z > 1e-6, z, 1e-6)

    L = np.zeros((2 * n, 6), dtype=np.float64)
    for i in range(n):
        x, y = pts[i]
        zi = z[i]
        L[2 * i, :] = [-1.0 / zi, 0.0, x / zi, x * y, -(1.0 + x * x), y]
        L[2 * i + 1, :] = [0.0, -1.0 / zi, y / zi, 1.0 + y * y, -x * y, -x]
    return L


class IBVSController:
    """Proportional image-based visual servoing controller.

    Parameters
    ----------
    kp
        Proportional gain. Applied uniformly to all six twist components;
        for asymmetric gains pass a ``(6,)`` array to :meth:`compute_velocity`
        directly via :attr:`kp_vec`.
    depth
        Default depth (metres) used to build the image Jacobian when no
        per-feature depth is supplied. The Jacobian degrades gracefully for
        modest depth errors, but you should override this when you have a
        live RGB-D measurement.
    damping
        Levenberg–Marquardt-style damping for the pseudo-inverse. Mitigates
        Jacobian singularities near degenerate feature configurations
        (e.g. all features collinear). Default ``0.0`` falls back to a plain
        ``np.linalg.pinv``.
    """

    def __init__(
        self,
        kp: float = 1.0,
        depth: float = 0.5,
        damping: float = 0.0,
    ) -> None:
        if depth <= 0.0:
            raise ValueError("depth must be positive")
        if damping < 0.0:
            raise ValueError("damping must be non-negative")
        self.kp = float(kp)
        self.depth = float(depth)
        self.damping = float(damping)
        # Optional per-axis gain override; None means use scalar ``kp``.
        self.kp_vec: np.ndarray | None = None

    # ------------------------------------------------------------------
    def set_gain(self, gain: float | np.ndarray) -> None:
        """Update the proportional gain.

        Accepts either a scalar (uniform gain) or a length-6 vector for
        per-axis gains in ``[v_x, v_y, v_z, ω_x, ω_y, ω_z]`` order.
        """
        if np.isscalar(gain):
            self.kp = float(gain)
            self.kp_vec = None
        else:
            vec = np.asarray(gain, dtype=np.float64).reshape(-1)
            if vec.size != 6:
                raise ValueError("Per-axis gain must have 6 elements")
            self.kp_vec = vec

    # ------------------------------------------------------------------
    def compute_velocity(
        self,
        s_current: np.ndarray,
        s_target: np.ndarray,
        depth: float | np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the body-frame twist that drives ``s_current → s_target``.

        Parameters
        ----------
        s_current, s_target
            Feature vectors of shape ``(2N,)`` or ``(N, 2)`` in normalised
            image coordinates. The pair must agree in feature count.
        depth
            Optional depth override (scalar or ``(N,)``). When ``None`` the
            controller's default depth is used.

        Returns
        -------
        np.ndarray
            Twist ``v ∈ R^6`` as ``[v_x, v_y, v_z, ω_x, ω_y, ω_z]``.
        """
        cur = np.asarray(s_current, dtype=np.float64).reshape(-1, 2)
        tgt = np.asarray(s_target, dtype=np.float64).reshape(-1, 2)
        if cur.shape != tgt.shape:
            raise ValueError(
                f"Feature shapes disagree: current={cur.shape}, target={tgt.shape}"
            )
        if cur.size == 0:
            return np.zeros(6, dtype=np.float64)

        L = build_image_jacobian(
            cur,
            depth=self.depth if depth is None else depth,
        )

        # Damped pseudo-inverse: (L^T L + μI)^{-1} L^T, falling back to the
        # plain Moore-Penrose pseudo-inverse when no damping is requested.
        if self.damping > 0.0:
            n_cols = L.shape[1]
            L_pinv = np.linalg.solve(
                L.T @ L + self.damping * np.eye(n_cols),
                L.T,
            )
        else:
            L_pinv = np.linalg.pinv(L)

        error = (tgt - cur).reshape(-1)  # s* - s
        # IBVS convention: velocity = -kp * L^+ * (s - s*) = +kp * L^+ * (s* - s).
        twist = L_pinv @ error
        if self.kp_vec is not None:
            twist = self.kp_vec * twist
        else:
            twist = self.kp * twist
        return twist.astype(np.float64)
