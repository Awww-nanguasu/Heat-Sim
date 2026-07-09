
"""
hybrid_gs_mionet.py

Hybrid Gauss-Seidel / MIONet iteration for

    A x = b

Main idea from the paper:

    ordinary step:
        x <- GS_step(A, b, x)

    every M steps:
        r = b - A x
        delta = MIONet(k, H(r), coords)
        x <- x + delta

This file intentionally keeps residual_to_sensor(...) as a user-provided callback,
because H(r) depends on how you convert the FEM residual vector into the same
fixed-size input used by the MIONet f/residual branch.

Recommended usage:
    A: scipy.sparse.csr_matrix
    b: numpy vector
    model: PyTorch MIONet model
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from scipy.sparse import csr_matrix, tril, triu, isspmatrix, isspmatrix_csr
from scipy.sparse.linalg import spsolve_triangular


@dataclass
class HybridInfo:
    converged: bool
    num_iter: int
    rel_res: float
    abs_res: float
    b_norm: float
    history: list[float]
    correction_iters: list[int]


def as_csr_matrix(A: Any) -> csr_matrix:
    if isspmatrix_csr(A):
        return A.astype(np.float64)
    if isspmatrix(A):
        return A.tocsr().astype(np.float64)
    return csr_matrix(np.asarray(A, dtype=np.float64))


def to_numpy_vector(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def gauss_seidel_sweep_with_scipy(
    DL: csr_matrix,
    U: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """One forward Gauss-Seidel sweep.

    A = D + L + U

    GS update:
        (D + L) x_new = b - U x_old

    This uses scipy.sparse.linalg.spsolve_triangular, i.e. "调包"完成一次 GS 扫描。
    """
    rhs = b - U @ x
    x_new = spsolve_triangular(DL, rhs, lower=True)
    return np.asarray(x_new, dtype=np.float64)


def mionet_correction(
    model: torch.nn.Module,
    k_sensor: np.ndarray,
    residual_sensor: np.ndarray,
    output_coords: np.ndarray,
    *,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> np.ndarray:
    """Evaluate MIONet correction.

    Expected shapes:
        k_sensor:        [k_dim] or [1, k_dim]
        residual_sensor: [f_dim] or [1, f_dim]
        output_coords:   [n_dofs, coord_dim]

    Returns:
        delta: [n_dofs]
    """
    model.eval()

    if k_sensor.ndim == 1:
        k_sensor = k_sensor[None, :]
    if residual_sensor.ndim == 1:
        residual_sensor = residual_sensor[None, :]

    k_t = torch.as_tensor(k_sensor, dtype=dtype, device=device)
    r_t = torch.as_tensor(residual_sensor, dtype=dtype, device=device)
    coords_t = torch.as_tensor(output_coords, dtype=dtype, device=device)

    with torch.no_grad():
        delta = model(k_t, r_t, coords_t)

    delta_np = delta.detach().cpu().numpy()

    # delta shape should be [1, n_dofs]
    return delta_np.reshape(-1).astype(np.float64)


def hybrid_gs_mionet(
    A: Any,
    b: Any,
    *,
    model: torch.nn.Module,
    k_sensor: np.ndarray,
    output_coords: np.ndarray,
    residual_to_sensor: Callable[[np.ndarray], np.ndarray],
    x0: Any | None = None,
    correction_period: int = 100,
    maxiter: int = 100000,
    rtol: float = 1e-10,
    atol: float = 0.0,
    check_every: int = 10,
    correction_scale: float = 1.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    verbose: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Hybrid GS-MIONet solver.

    Args:
        A, b:
            Linear system A x = b.
        model:
            Pretrained MIONet. Must accept model(k_sensor, residual_sensor, output_coords).
        k_sensor:
            Fixed branch input for k. Shape [k_dim].
        output_coords:
            Coordinates where MIONet outputs correction. Usually FEM dof/node coords.
            Shape [n_dofs, coord_dim].
        residual_to_sensor:
            Function converting residual vector r = b - A x into MIONet residual branch input.
            It must return shape [f_dim], matching training f/residual input dimension.
        x0:
            Initial guess. If None, zeros are used.
        correction_period:
            Apply MIONet correction every M GS sweeps.
        correction_scale:
            Optional damping for MIONet correction:
                x <- x + correction_scale * delta
            Use 1.0 to match the paper. Use 0.5 etc. if unstable.
    """
    if correction_period <= 0:
        raise ValueError("correction_period must be positive")

    A = as_csr_matrix(A)
    b = to_numpy_vector(b)

    n, m = A.shape
    if n != m:
        raise ValueError(f"A must be square, got {A.shape}")
    if b.shape[0] != n:
        raise ValueError(f"b length {b.shape[0]} does not match A shape {A.shape}")

    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
    else:
        x = to_numpy_vector(x0).copy()

    if output_coords.shape[0] != n:
        raise ValueError(
            f"output_coords first dimension should match n={n}, "
            f"got {output_coords.shape}"
        )

    # A = DL + U, with DL = D + L.
    DL = tril(A, k=0, format="csr").astype(np.float64)
    U = triu(A, k=1, format="csr").astype(np.float64)

    b_norm = np.linalg.norm(b)
    if b_norm == 0:
        b_norm = 1.0

    history: list[float] = []
    correction_iters: list[int] = []

    r = b - A @ x
    abs_res = np.linalg.norm(r)
    rel_res = abs_res / b_norm
    history.append(rel_res)

    if verbose:
        print(f"iter=0, rel_res={rel_res:.6e}, abs_res={abs_res:.6e}")

    if abs_res <= atol or rel_res <= rtol:
        info = HybridInfo(True, 0, rel_res, abs_res, b_norm, history, correction_iters)
        return x, info.__dict__

    converged = False
    last_iter = 0

    for it in range(1, maxiter + 1):
        # 1. ordinary GS step
        x = gauss_seidel_sweep_with_scipy(DL, U, b, x)

        # 2. MIONet correction every M sweeps
        if it % correction_period == 0:
            r = b - A @ x
            residual_sensor = residual_to_sensor(r)

            delta = mionet_correction(
                model=model,
                k_sensor=k_sensor,
                residual_sensor=residual_sensor,
                output_coords=output_coords,
                device=device,
                dtype=dtype,
            )

            if delta.shape[0] != n:
                raise ValueError(
                    f"MIONet correction has length {delta.shape[0]}, expected {n}"
                )

            x = x + correction_scale * delta
            correction_iters.append(it)

        # 3. residual check
        if it % check_every == 0 or it == maxiter or it % correction_period == 0:
            r = b - A @ x
            abs_res = np.linalg.norm(r)
            rel_res = abs_res / b_norm
            history.append(rel_res)

            if verbose:
                tag = " corrected" if it in correction_iters else ""
                print(f"iter={it}{tag}, rel_res={rel_res:.6e}, abs_res={abs_res:.6e}")

            if abs_res <= atol or rel_res <= rtol:
                converged = True
                last_iter = it
                break

        last_iter = it

    info = HybridInfo(
        converged=converged,
        num_iter=last_iter,
        rel_res=rel_res,
        abs_res=abs_res,
        b_norm=b_norm,
        history=history,
        correction_iters=correction_iters,
    )
    return x, info.__dict__


def identity_residual_to_sensor(r: np.ndarray) -> np.ndarray:
    """Only valid when your MIONet residual branch dimension equals len(r).

    In the paper, this is generally NOT enough. They convert residual vector r
    to residual function H(r), then sample it on fixed sensor points.
    """
    return np.asarray(r, dtype=np.float64).reshape(-1)
