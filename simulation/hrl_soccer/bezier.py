from __future__ import annotations

import math
import numpy as np


def bernstein_basis(order: int, t: np.ndarray) -> np.ndarray:
    """Return Bernstein basis values for an order-n Bezier curve."""
    t = np.asarray(t, dtype=np.float32)
    terms = []
    for i in range(order + 1):
        coeff = math.comb(order, i)
        terms.append(coeff * np.power(1.0 - t, order - i) * np.power(t, i))
    return np.stack(terms, axis=-1).astype(np.float32)


def bezier(control_points: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    """Evaluate a Bezier curve.

    Args:
        control_points: shape (dims, n_points)
        t: scalar or array in [0, 1]
    """
    cp = np.asarray(control_points, dtype=np.float32)
    if cp.ndim != 2:
        raise ValueError(f"control_points must be 2D, got {cp.shape}")
    t_arr = np.asarray(t, dtype=np.float32)
    scalar = t_arr.ndim == 0
    t_arr = np.atleast_1d(t_arr)
    basis = bernstein_basis(cp.shape[1] - 1, np.clip(t_arr, 0.0, 1.0))
    out = basis @ cp.T
    if scalar:
        return out[0]
    return out


def default_kick_curve() -> np.ndarray:
    """Nominal front-right toe path used as the reference kicking primitive."""
    return np.array(
        [
            [0.18, 0.20, 0.12, 0.38, 0.54],
            [-0.14, -0.18, -0.22, -0.12, -0.04],
            [-0.31, -0.12, -0.05, -0.12, -0.28],
        ],
        dtype=np.float32,
    )


def sample_kick_curve(rng: np.random.Generator) -> np.ndarray:
    """Sample a randomized 3x5 Bezier toe trajectory."""
    cp = default_kick_curve()
    cp += rng.normal(0.0, np.array([[0.04], [0.04], [0.03]], dtype=np.float32), cp.shape)
    cp[0] = np.clip(cp[0], 0.02, 0.75)
    cp[1] = np.clip(cp[1], -0.40, 0.20)
    cp[2] = np.clip(cp[2], -0.45, -0.02)
    return cp.astype(np.float32)
