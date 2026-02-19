from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
from tqdm import tqdm


def integrate_em(
    system,
    y0: Optional[Iterable[float]] = None,
    t_span: Tuple[float, float] = (0.0, 1.0),
    dt: float = 0.01,
    tau: int = 1,
    transient: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Euler-Maruyama integration for a system with drift and diffusion.

    The system is expected to implement:
        - drift(t, y) -> array-like (dim,)
        - diffusion(t, y) -> array-like (dim, dim) or (dim,)

    Parameters:
        system: object providing drift and diffusion.
        y0: initial condition. If None, uses system.default_y0 if present.
        t_span: (t0, tf)
        dt: time step
        tau: save every tau steps
        transient: discard samples with t < transient
        rng: numpy Generator
        show_progress: show tqdm progress bar

    Returns:
        tsave: time grid (after transient cut)
        ysave: trajectory samples (after transient cut)
    """
    if y0 is None:
        if not hasattr(system, "default_y0"):
            raise ValueError("y0 must be provided if system has no default_y0")
        y0 = getattr(system, "default_y0")

    yold = np.asarray(y0, dtype=float)

    t0, tf = t_span
    n_steps = int((tf - t0) / dt)
    if n_steps <= 0:
        raise ValueError("t_span and dt produce no integration steps")

    if tau < 1:
        raise ValueError("tau must be >= 1")

    n_saves = n_steps // tau
    tsave = t0 + np.arange(n_saves) * (tau * dt)
    ysave = np.zeros((len(tsave), yold.shape[0]))

    if rng is None:
        rng = np.random.default_rng()

    index = 0
    for i in tqdm(range(n_steps), disable=not show_progress):
        t = t0 + i * dt
        f = system.drift(t=t, y=yold)
        g = system.diffusion(t=t, y=yold)
        dW = rng.normal(0, np.sqrt(dt), size=yold.shape[0])

        g = np.asarray(g)
        if g.ndim == 2:
            ynew = yold + f * dt + g @ dW
        else:
            ynew = yold + f * dt + g * dW

        if np.mod(i, tau) == 0:
            ysave[index, :] = ynew
            index += 1

        yold = ynew.copy()

    if transient <= t0:
        return tsave, ysave

    valid = np.where(tsave >= transient)[0]
    if valid.size == 0:
        return np.array([]), np.empty((0, ysave.shape[1]))

    ind_transient = valid[0]
    return tsave[ind_transient:], ysave[ind_transient:, :]
