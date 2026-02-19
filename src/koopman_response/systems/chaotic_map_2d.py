from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

TWO_PI = 2 * np.pi


@dataclass
class NoisyChaoticMap2D:
    """
    2D chaotic map:

        [x_{n+1}, y_{n+1}]^T = [A (x_n, y_n)^T
            + (1/pi) * zeta(x_n + y_n; mu) * (1, 1)^T
            + sigma * eta] mod 1

    where A = [[2, 1], [1, 1]],
    zeta(s) = arctan( |mu| sin(2π s - alpha) / (1 - |mu| cos(2π s - alpha)) ),
    mu = |mu| e^{i alpha}, and eta ~ N(0, I).
    """

    mu_abs: float = 0.88
    alpha: float = -2.4
    sigma: float = 0.01
    A: np.ndarray = field(default_factory=lambda: np.array([[2.0, 1.0], [1.0, 1.0]]))

    def zeta(self, s: np.ndarray) -> np.ndarray:
        numerator = self.mu_abs * np.sin(TWO_PI * s - self.alpha)
        denominator = 1.0 - self.mu_abs * np.cos(TWO_PI * s - self.alpha)
        return np.arctan(numerator / denominator)

    def step(self, xy: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        xy = np.asarray(xy, dtype=float).reshape(2)
        if rng is None:
            rng = np.random.default_rng()

        s = xy[0] + xy[1]
        z = self.zeta(s)
        noise = self.sigma * rng.normal(0.0, 1.0, size=2)

        updated = self.A @ xy + (1.0 / np.pi) * z * np.ones(2) + noise
        return np.mod(updated, 1.0)

    def iterate(
        self,
        x0: Tuple[float, float],
        n_steps: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if rng is None:
            rng = np.random.default_rng()

        xs = np.empty((n_steps + 1, 2), dtype=float)
        xs[0, :] = np.asarray(x0, dtype=float)
        x = xs[0]
        for i in range(1, n_steps + 1):
            x = self.step(x, rng=rng)
            xs[i, :] = x
        return xs
