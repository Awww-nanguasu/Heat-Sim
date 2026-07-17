
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr, isspmatrix


@dataclass
class GSInfo:
    converged: bool
    num_iter: int
    rel_res: float
    abs_res: float
    b_norm: float
    history: list[float]


def _to_numpy_vector(b: Any) -> np.ndarray:
    """Convert vector-like input to a numpy float64 vector."""
    if hasattr(b, "detach"):  # torch tensor
        b = b.detach().cpu().numpy()
    else:
        b = np.asarray(b)

    return np.asarray(b, dtype=np.float64).reshape(-1)


def torch_sparse_to_scipy_csr(A: Any) -> csr_matrix:
    """Convert torch dense/sparse tensor to scipy CSR."""
    import torch

    if not isinstance(A, torch.Tensor):
        raise TypeError("A is not a torch.Tensor")

    if A.layout == torch.sparse_coo:
        A = A.coalesce()
        idx = A.indices().detach().cpu().numpy()
        val = A.values().detach().cpu().numpy().astype(np.float64)
        return csr_matrix((val, (idx[0], idx[1])), shape=tuple(A.shape))

    if A.layout == torch.sparse_csr:
        crow = A.crow_indices().detach().cpu().numpy()
        col = A.col_indices().detach().cpu().numpy()
        val = A.values().detach().cpu().numpy().astype(np.float64)
        return csr_matrix((val, col, crow), shape=tuple(A.shape))

    return csr_matrix(A.detach().cpu().numpy().astype(np.float64))


def as_csr_matrix(A: Any) -> csr_matrix:
    """Convert scipy / torch / numpy matrix to scipy CSR."""
    if isspmatrix_csr(A):
        return A.astype(np.float64)

    if isspmatrix(A):
        return A.tocsr().astype(np.float64)

    if hasattr(A, "layout"):  # likely torch tensor
        return torch_sparse_to_scipy_csr(A)

    return csr_matrix(np.asarray(A, dtype=np.float64))


def gauss_seidel(
    A: Any,
    b: Any,
    x0: Any | None = None,
    *,
    maxiter: int = 10000,
    rtol: float = 1e-10,
    atol: float = 0.0,
    omega: float = 1.0,
    check_every: int = 10,
    return_history: bool = True,
    verbose: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve A x = b using Gauss-Seidel or SOR.

    Args:
        A: System matrix. Recommended: scipy.sparse.csr_matrix.
        b: Right-hand side vector.
        x0: Initial guess. If None, zeros are used.
        maxiter: Maximum number of sweeps.
        rtol: Relative residual tolerance.
        atol: Absolute residual tolerance.
        omega:
            omega = 1.0 gives standard Gauss-Seidel.
            0 < omega < 1 gives under-relaxation.
            1 < omega < 2 gives SOR.
        check_every: Compute residual every this many iterations.
        return_history: Save relative residual history.
        verbose: Print progress.

    Returns:
        x, info
    """
    if not (0.0 < omega < 2.0):
        raise ValueError("omega must satisfy 0 < omega < 2")

    A = as_csr_matrix(A)
    b = _to_numpy_vector(b)

    n, m = A.shape
    if n != m:
        raise ValueError(f"A must be square, got shape {A.shape}")
    if b.shape[0] != n:
        raise ValueError(f"b length {b.shape[0]} does not match A shape {A.shape}")

    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
    else:
        x = _to_numpy_vector(x0).copy()
        if x.shape[0] != n:
            raise ValueError(f"x0 length {x.shape[0]} does not match A shape {A.shape}")

    indptr = A.indptr
    indices = A.indices
    data = A.data

    diag = A.diagonal().astype(np.float64)
    if np.any(diag == 0.0):
        zero_ids = np.where(diag == 0.0)[0][:10]
        raise ZeroDivisionError(f"A has zero diagonal entries, examples: {zero_ids}")

    b_norm = np.linalg.norm(b)
    if b_norm == 0.0:
        b_norm = 1.0

    history: list[float] = []

    r = b - A @ x
    abs_res = np.linalg.norm(r)
    rel_res = abs_res / b_norm

    if return_history:
        history.append(rel_res)

    if verbose:
        print(f"iter=0, rel_res={rel_res:.6e}, abs_res={abs_res:.6e}")

    if abs_res <= atol or rel_res <= rtol:
        info = GSInfo(True, 0, rel_res, abs_res, b_norm, history)
        return x, info.__dict__

    converged = False
    last_iter = 0

    for it in range(1, maxiter + 1):
        for i in range(n):
            row_start = indptr[i]
            row_end = indptr[i + 1]

            aii = diag[i]
            sigma = 0.0

            for p in range(row_start, row_end):
                j = indices[p]
                if j != i:
                    sigma += data[p] * x[j]

            x_gs = (b[i] - sigma) / aii
            x[i] = (1.0 - omega) * x[i] + omega * x_gs

        last_iter = it

        if it % check_every == 0 or it == maxiter:
            r = b - A @ x
            abs_res = np.linalg.norm(r)
            rel_res = abs_res / b_norm

            if return_history:
                history.append(rel_res)

            if verbose:
                print(f"iter={it}, rel_res={rel_res:.6e}, abs_res={abs_res:.6e}")

            if abs_res <= atol or rel_res <= rtol:
                converged = True
                break

    info = GSInfo(converged, last_iter, rel_res, abs_res, b_norm, history)
    return x, info.__dict__


def residual_norm(A: Any, x: Any, b: Any) -> tuple[float, float]:
    """Return absolute and relative residual norm of A x = b."""
    A = as_csr_matrix(A)
    x = _to_numpy_vector(x)
    b = _to_numpy_vector(b)

    r = b - A @ x
    abs_res = np.linalg.norm(r)
    b_norm = np.linalg.norm(b)
    if b_norm == 0.0:
        b_norm = 1.0
    return abs_res, abs_res / b_norm


if __name__ == "__main__":
    from scipy.sparse import diags

    n = 100
    A = diags(
        diagonals=[-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)],
        offsets=[-1, 0, 1],
        format="csr",
    )
    b = np.ones(n)

    x, info = gauss_seidel(A, b, maxiter=10000, rtol=1e-10, check_every=100)
    print(info)
