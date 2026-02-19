"""
Lorenz 63 system (with additive noise parameter).
"""

from __future__ import annotations

import numpy as np


class NoisyLorenz63:
    def __init__(
        self,
        rho: float = 28.0,
        sigma: float = 10.0,
        beta: float = 8.0 / 3.0,
        noise: float = 2.0,
    ):
        self.rho = rho
        self.sigma = sigma
        self.beta = beta
        self.noise = noise

        self.default_y0 = np.array([1.0, 0.5, 2.0])

    def drift(self, t: float, y: np.ndarray) -> np.ndarray:
        _ = t
        sigma = self.sigma
        rho = self.rho
        beta = self.beta
        x, yv, z = y
        return np.array([sigma * (yv - x), x * (rho - z) - yv, x * yv - beta * z])

    def diffusion(self, t: float, y: np.ndarray) -> np.ndarray:
        _ = t
        _ = y
        return self.noise * np.eye(3)
