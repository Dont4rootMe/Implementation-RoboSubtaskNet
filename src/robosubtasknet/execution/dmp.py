"""Dynamic Movement Primitives (DMPs) for sub-task motion playback.

Phase 2 of the RoboSubtaskNet pipeline (Section 12 of the implementation plan)
maps each predicted sub-task label onto a parameterized motion primitive. The
most standard choice — and the one we use here — is the *discrete DMP*
formulation of Ijspeert et al. (2013): a critically-damped second-order
attractor toward a goal, modulated by a non-linear forcing term learned from
demonstration.

This module provides a self-contained minimal implementation. We intentionally
do not depend on ``pydmps`` so the package stays lean; ``pydmps`` (and the more
production-grade ``movement_primitives`` from DLR) are the obvious upgrades if
richer features (online adaptation, coupling terms, obstacle avoidance) are
later required.

References
----------
Ijspeert, A. J. et al. *Dynamical Movement Primitives: Learning Attractor
Models for Motor Behaviors.* Neural Computation, 25(2):328-373, 2013.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["DiscreteDMP", "MultiDOFDMP", "SubtaskDMPLibrary"]


# ---------------------------------------------------------------------------
# Single-DOF discrete DMP
# ---------------------------------------------------------------------------


class DiscreteDMP:
    """Discrete dynamical movement primitive in one degree of freedom.

    The transformation system is the classic second-order attractor

    .. math::
        \\tau \\dot z = \\alpha_z (\\beta_z (g - y) - z) + f(x)
        \\tau \\dot y = z

    and the canonical phase variable obeys

    .. math::
        \\tau \\dot x = -\\alpha_x x,\\quad x(0) = 1, x \\to 0.

    The forcing term :math:`f(x)` is a weighted sum of Gaussian basis
    functions over the phase :math:`x \\in [1, 0]`, scaled by ``x * (g - y0)``
    so the forcing decays as the system converges and respects spatial
    re-scaling between training and rollout goals.

    Parameters
    ----------
    n_basis
        Number of Gaussian basis functions placed along the phase.
    alpha_z, beta_z
        Spring-damper constants of the transformation system. Default
        ``alpha_z = 25, beta_z = alpha_z / 4 = 6.25`` gives critical damping.
    alpha_x
        Decay rate of the canonical phase. Smaller values make the primitive
        last longer in phase space (slower forcing).
    """

    def __init__(
        self,
        n_basis: int = 25,
        alpha_z: float = 25.0,
        beta_z: float = 6.25,
        alpha_x: float = 1.0,
    ) -> None:
        if n_basis < 1:
            raise ValueError("n_basis must be >= 1")
        self.n_basis = int(n_basis)
        self.alpha_z = float(alpha_z)
        self.beta_z = float(beta_z)
        self.alpha_x = float(alpha_x)

        # Basis centers spaced uniformly in time then mapped to phase space.
        # t_i in [0, 1] -> x_i = exp(-alpha_x * t_i).
        t_centers = np.linspace(0.0, 1.0, self.n_basis)
        self.centers = np.exp(-self.alpha_x * t_centers)
        # Widths chosen as in Ijspeert et al. so basis functions overlap.
        # h_i = 1 / (c_{i+1} - c_i)^2; for the last center reuse the prior width.
        diffs = np.diff(self.centers)
        widths = 1.0 / (diffs ** 2 + 1e-8)
        self.widths = np.concatenate([widths, widths[-1:]])

        # Learned forcing weights — zero-initialised so the unlearnt DMP is
        # a pure spring-damper to the goal.
        self.weights = np.zeros(self.n_basis, dtype=np.float64)

        # Spatial anchors captured at fit time and re-used at rollout if the
        # caller does not supply explicit ``y0`` / ``goal``.
        self.y0: float = 0.0
        self.goal: float = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _basis(self, x: np.ndarray | float) -> np.ndarray:
        """Evaluate all basis functions at one or more phase samples."""
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        # psi has shape [len(x), n_basis]
        return np.exp(-self.widths[None, :] * (x[:, None] - self.centers[None, :]) ** 2)

    def _phase_trajectory(self, n_steps: int, dt: float, tau: float) -> np.ndarray:
        """Integrate the canonical system, returning the phase at each step."""
        x = np.empty(n_steps, dtype=np.float64)
        x_val = 1.0
        for i in range(n_steps):
            x[i] = x_val
            # Euler step: tau * dx = -alpha_x * x  ->  x += -alpha_x/tau * x * dt
            x_val = max(x_val + (-self.alpha_x / max(tau, 1e-8)) * x_val * dt, 0.0)
        return x

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def learn_from_demo(self, y_demo: np.ndarray, dt: float) -> None:
        """Fit the forcing weights to reproduce ``y_demo`` under the DMP.

        Uses locally weighted regression (one regression per basis function)
        on the demonstrated forcing signal obtained by inverting the
        transformation system. The integration assumes ``tau = 1`` during
        learning; ``tau`` only re-scales the rollout (Section 2.4 of
        Ijspeert et al.).
        """
        y_demo = np.asarray(y_demo, dtype=np.float64).flatten()
        if y_demo.size < 3:
            raise ValueError("Need at least three demonstration samples to fit a DMP")
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        # Finite-difference velocity and acceleration.
        dy = np.gradient(y_demo, dt)
        ddy = np.gradient(dy, dt)

        self.y0 = float(y_demo[0])
        self.goal = float(y_demo[-1])

        # Phase trajectory with tau = 1.
        x_traj = self._phase_trajectory(n_steps=y_demo.size, dt=dt, tau=1.0)

        # Target forcing implied by the transformation system, tau = 1.
        # f_target(t) = ddy - alpha_z * (beta_z * (g - y) - dy)
        f_target = ddy - self.alpha_z * (self.beta_z * (self.goal - y_demo) - dy)

        # Locally weighted regression per basis (Schaal & Atkeson style).
        psi = self._basis(x_traj)  # [T, n_basis]
        s = x_traj * (self.goal - self.y0)  # scaling factor for each step
        denom = np.einsum("t,tn,t->n", s, psi, s) + 1e-10
        numer = np.einsum("t,tn,t->n", s, psi, f_target)
        self.weights = numer / denom

    def rollout(
        self,
        y0: float | None = None,
        goal: float | None = None,
        dt: float = 0.01,
        tau: float = 1.0,
    ) -> np.ndarray:
        """Integrate the DMP forward and return the position trajectory.

        Parameters
        ----------
        y0, goal
            Start and goal positions. Falls back to the values captured during
            ``learn_from_demo`` if omitted.
        dt
            Integration step.
        tau
            Temporal scaling: ``tau > 1`` slows the motion, ``tau < 1``
            speeds it up. The integration horizon is ``ceil(tau / dt)``.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if tau <= 0.0:
            raise ValueError("tau must be positive")

        if y0 is None:
            y0 = self.y0
        if goal is None:
            goal = self.goal
        y0 = float(y0)
        goal = float(goal)

        n_steps = max(int(np.ceil(tau / dt)), 2)
        traj = np.empty(n_steps, dtype=np.float64)
        y = y0
        z = 0.0  # velocity-like state, z = tau * dy
        x = 1.0
        for i in range(n_steps):
            traj[i] = y
            # Forcing term.
            psi = self._basis(x)[0]
            num = float(np.dot(psi, self.weights))
            den = float(psi.sum()) + 1e-10
            f = (num / den) * x * (goal - y0)
            # Transformation system, Euler integration with tau scaling.
            dz = (self.alpha_z * (self.beta_z * (goal - y) - z) + f) / tau
            dy = z / tau
            z = z + dz * dt
            y = y + dy * dt
            # Canonical system.
            x = max(x + (-self.alpha_x / tau) * x * dt, 0.0)
        return traj


# ---------------------------------------------------------------------------
# Multi-DOF wrapper
# ---------------------------------------------------------------------------


class MultiDOFDMP:
    """Independent per-dimension DMPs sharing canonical-system parameters.

    The classical formulation runs one transformation system per dimension
    with a common phase variable; here we keep them fully decoupled which is
    equivalent up to the phase-synchronisation choice (each ``DiscreteDMP``
    drives its own phase from the same initial conditions and the same
    ``alpha_x``, so they stay aligned in expectation).

    Parameters
    ----------
    n_dims
        Number of independent dimensions (e.g. 3 for Cartesian xyz, 6 for a
        SE(3) end-effector twist, 7 for joint space on a 7-DOF arm).
    All remaining kwargs are forwarded to :class:`DiscreteDMP`.
    """

    def __init__(
        self,
        n_dims: int,
        n_basis: int = 25,
        alpha_z: float = 25.0,
        beta_z: float = 6.25,
        alpha_x: float = 1.0,
    ) -> None:
        if n_dims < 1:
            raise ValueError("n_dims must be >= 1")
        self.n_dims = int(n_dims)
        self.dmps = [
            DiscreteDMP(
                n_basis=n_basis,
                alpha_z=alpha_z,
                beta_z=beta_z,
                alpha_x=alpha_x,
            )
            for _ in range(self.n_dims)
        ]

    def learn_from_demo(self, y_demo: np.ndarray, dt: float) -> None:
        """Fit each per-dimension DMP independently.

        Parameters
        ----------
        y_demo
            Demonstration trajectory of shape ``[T, n_dims]``.
        dt
            Sampling period of the demonstration.
        """
        y_demo = np.asarray(y_demo, dtype=np.float64)
        if y_demo.ndim != 2 or y_demo.shape[1] != self.n_dims:
            raise ValueError(
                f"Expected demo shape [T, {self.n_dims}], got {y_demo.shape}"
            )
        for d, dmp in enumerate(self.dmps):
            dmp.learn_from_demo(y_demo[:, d], dt=dt)

    def rollout(
        self,
        y0: np.ndarray | None = None,
        goal: np.ndarray | None = None,
        dt: float = 0.01,
        tau: float = 1.0,
    ) -> np.ndarray:
        """Integrate every DOF and stack into a ``[T, n_dims]`` trajectory."""
        if y0 is not None:
            y0 = np.asarray(y0, dtype=np.float64).reshape(self.n_dims)
        if goal is not None:
            goal = np.asarray(goal, dtype=np.float64).reshape(self.n_dims)

        per_dim = []
        for d, dmp in enumerate(self.dmps):
            y0_d = None if y0 is None else float(y0[d])
            goal_d = None if goal is None else float(goal[d])
            per_dim.append(dmp.rollout(y0=y0_d, goal=goal_d, dt=dt, tau=tau))
        # All per-dim trajectories share the same length because they share dt
        # and tau, but we trim defensively in case of rounding.
        n = min(len(t) for t in per_dim)
        return np.stack([t[:n] for t in per_dim], axis=1)


# ---------------------------------------------------------------------------
# Sub-task -> DMP library
# ---------------------------------------------------------------------------


@dataclass
class _LibraryEntry:
    """Bookkeeping for a single fitted primitive in the library."""

    dmp: MultiDOFDMP
    n_dims: int
    n_demos: int = 0
    dt: float = 0.0


@dataclass
class SubtaskDMPLibrary:
    """Map sub-task name to a fitted :class:`MultiDOFDMP`.

    Typical usage::

        lib = SubtaskDMPLibrary()
        lib.fit("reach", demos_for_reach)        # list of [T_i, n_dims] arrays
        lib.fit("pick", demos_for_pick)
        traj = lib.execute("reach", current_state=p_now, goal=p_target, dt=0.01)

    Parameters
    ----------
    n_basis, alpha_z, beta_z, alpha_x
        Forwarded to each newly constructed primitive.
    """

    n_basis: int = 25
    alpha_z: float = 25.0
    beta_z: float = 6.25
    alpha_x: float = 1.0
    _entries: dict[str, _LibraryEntry] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def names(self) -> list[str]:
        """Return the sub-task names currently registered."""
        return list(self._entries.keys())

    # ------------------------------------------------------------------
    def fit(self, name: str, demonstrations: list[np.ndarray]) -> None:
        """Fit a multi-DOF DMP for ``name`` by averaging per-demo forcing weights.

        Parameters
        ----------
        name
            Sub-task label (e.g. ``"reach"``, ``"pick"``).
        demonstrations
            Non-empty list of arrays of shape ``[T_i, n_dims]``. All
            demonstrations must share ``n_dims``; ``T_i`` may differ.

        Notes
        -----
        We fit one DMP per demonstration and then average the learned forcing
        weights — a simple, robust alternative to stacking all demonstrations
        into one big regression. For a single demonstration this reduces to
        plain LWR. For multiple demonstrations it is a reasonable mean primitive;
        for richer behaviour (PMPs, GMM-based DMPs) see ``movement_primitives``.
        """
        if not demonstrations:
            raise ValueError(f"No demonstrations provided for sub-task '{name}'")

        n_dims = np.asarray(demonstrations[0]).reshape(len(demonstrations[0]), -1).shape[1]
        for i, demo in enumerate(demonstrations):
            arr = np.asarray(demo)
            if arr.ndim != 2 or arr.shape[1] != n_dims:
                raise ValueError(
                    f"Demo {i} for '{name}' has shape {arr.shape}; expected [T, {n_dims}]"
                )

        # Use unit dt during averaging — this only matters for the velocity
        # finite differences and is consistent across all demos here.
        dt = 1.0 / max(len(demonstrations[0]) - 1, 1)

        accum_weights: list[np.ndarray] = [
            np.zeros(self.n_basis, dtype=np.float64) for _ in range(n_dims)
        ]
        y0_accum = np.zeros(n_dims, dtype=np.float64)
        goal_accum = np.zeros(n_dims, dtype=np.float64)
        for demo in demonstrations:
            mdmp = MultiDOFDMP(
                n_dims=n_dims,
                n_basis=self.n_basis,
                alpha_z=self.alpha_z,
                beta_z=self.beta_z,
                alpha_x=self.alpha_x,
            )
            mdmp.learn_from_demo(demo, dt=dt)
            for d in range(n_dims):
                accum_weights[d] += mdmp.dmps[d].weights
                y0_accum[d] += mdmp.dmps[d].y0
                goal_accum[d] += mdmp.dmps[d].goal

        n = float(len(demonstrations))
        mean_dmp = MultiDOFDMP(
            n_dims=n_dims,
            n_basis=self.n_basis,
            alpha_z=self.alpha_z,
            beta_z=self.beta_z,
            alpha_x=self.alpha_x,
        )
        for d in range(n_dims):
            mean_dmp.dmps[d].weights = accum_weights[d] / n
            mean_dmp.dmps[d].y0 = float(y0_accum[d] / n)
            mean_dmp.dmps[d].goal = float(goal_accum[d] / n)
        self._entries[name] = _LibraryEntry(
            dmp=mean_dmp,
            n_dims=n_dims,
            n_demos=len(demonstrations),
            dt=dt,
        )

    # ------------------------------------------------------------------
    def execute(
        self,
        name: str,
        current_state: np.ndarray,
        goal: np.ndarray,
        dt: float,
        tau: float = 1.0,
    ) -> np.ndarray:
        """Roll out the named primitive from ``current_state`` to ``goal``.

        Parameters
        ----------
        name
            Sub-task label previously registered via :meth:`fit`.
        current_state, goal
            Per-dimension start/goal vectors of shape ``[n_dims]``.
        dt
            Integration step (seconds).
        tau
            Temporal scaling factor passed to the underlying DMP.

        Returns
        -------
        np.ndarray
            Trajectory of shape ``[T_out, n_dims]``.
        """
        if name not in self._entries:
            raise KeyError(
                f"Sub-task '{name}' is not in the library; "
                f"known sub-tasks: {self.names()}"
            )
        entry = self._entries[name]
        current_state = np.asarray(current_state, dtype=np.float64).reshape(-1)
        goal = np.asarray(goal, dtype=np.float64).reshape(-1)
        if current_state.size != entry.n_dims or goal.size != entry.n_dims:
            raise ValueError(
                f"Sub-task '{name}' expects {entry.n_dims}-D states; got "
                f"current={current_state.shape}, goal={goal.shape}"
            )
        return entry.dmp.rollout(y0=current_state, goal=goal, dt=dt, tau=tau)
