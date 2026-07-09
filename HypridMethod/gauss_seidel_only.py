
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import time
import numpy as np
from scipy.sparse import csr_matrix, tril, triu, isspmatrix, isspmatrix_csr
from scipy.sparse.linalg import spsolve_triangular


@dataclass
class GSInfo:
    converged: bool
    num_iter: int
    rel_res: float
    abs_res: float
    b_norm: float
    history: list[float]
    elapsed_time: float


def _to_numpy_array(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        try:
            return np.asarray(x.numpy())
        except Exception:
            pass
    return np.asarray(x)


def to_numpy_vector(x: Any) -> np.ndarray:
    return _to_numpy_array(x).astype(np.float64).reshape(-1)


def _get_attr_or_method(obj: Any, names: list[str]) -> Any | None:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if callable(value):
                try:
                    return value()
                except TypeError:
                    continue
            return value
    return None


def as_csr_matrix(A: Any) -> csr_matrix:
    """Convert scipy sparse / FEALPy CSRTensor-like / dense matrix to scipy CSR."""
    if isspmatrix_csr(A):
        return A.astype(np.float64)

    if isspmatrix(A):
        return A.tocsr().astype(np.float64)

    for method_name in [
        "to_scipy",
        "to_scipy_csr",
        "to_scipy_sparse",
        "to_scipy_matrix",
        "to_csr_matrix",
        "scipy_csr",
    ]:
        if hasattr(A, method_name):
            method = getattr(A, method_name)
            if callable(method):
                try:
                    B = method()
                    if isspmatrix(B):
                        return B.tocsr().astype(np.float64)
                except Exception:
                    pass

    data = _get_attr_or_method(A, ["data", "values", "_data", "_values"])
    indices = _get_attr_or_method(A, ["indices", "col", "cols", "col_indices", "_indices", "_col"])
    indptr = _get_attr_or_method(A, ["indptr", "crow", "crow_indices", "rowptr", "_indptr", "_crow"])

    if data is not None and indices is not None and indptr is not None:
        data_np = _to_numpy_array(data).astype(np.float64)
        indices_np = _to_numpy_array(indices).astype(np.int64)
        indptr_np = _to_numpy_array(indptr).astype(np.int64)

        shape = getattr(A, "shape", None)
        if callable(shape):
            shape = shape()

        if shape is None:
            shape = getattr(A, "sparse_shape", None)
            if callable(shape):
                shape = shape()

        if shape is None:
            nrow = len(indptr_np) - 1
            ncol = int(indices_np.max()) + 1 if len(indices_np) > 0 else nrow
            shape = (nrow, ncol)
        else:
            shape = tuple(shape)

        return csr_matrix((data_np, indices_np, indptr_np), shape=shape)

    return csr_matrix(np.asarray(A, dtype=np.float64))


def debug_sparse_object(A: Any) -> None:
    print("type(A):", type(A))
    print("shape:", getattr(A, "shape", None))
    print("sparse_shape:", getattr(A, "sparse_shape", None))

    useful_names = [
        "to_scipy", "to_scipy_csr", "to_scipy_sparse", "to_scipy_matrix",
        "to_csr_matrix", "scipy_csr",
        "data", "values", "_data", "_values",
        "indices", "col", "cols", "col_indices", "_indices", "_col",
        "indptr", "crow", "crow_indices", "rowptr", "_indptr", "_crow",
    ]

    for name in useful_names:
        if hasattr(A, name):
            value = getattr(A, name)
            print(f"has {name}: type={type(value)}, callable={callable(value)}")


def gauss_seidel_sweep_with_scipy(
    DL: csr_matrix,
    U: csr_matrix,
    b: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """One forward GS sweep: (D + L) x_new = b - U x_old."""
    rhs = b - U @ x
    x_new = spsolve_triangular(DL, rhs, lower=True)
    return np.asarray(x_new, dtype=np.float64)


def gauss_seidel_only(
    A: Any,
    b: Any,
    *,
    x0: Any | None = None,
    maxiter: int = 100000,
    rtol: float = 1.0e-10,
    atol: float = 0.0,
    check_every: int = 10,
    verbose: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Pure Gauss-Seidel solver for A x = b."""
    t0 = time.time()

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
        if x.shape[0] != n:
            raise ValueError(f"x0 length {x.shape[0]} does not match A shape {A.shape}")

    DL = tril(A, k=0, format="csr").astype(np.float64)
    U = triu(A, k=1, format="csr").astype(np.float64)

    diag = DL.diagonal()
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
    history.append(rel_res)

    if verbose:
        print(f"GS iter=0, rel_res={rel_res:.6e}, abs_res={abs_res:.6e}")

    if abs_res <= atol or rel_res <= rtol:
        info = GSInfo(True, 0, rel_res, abs_res, b_norm, history, time.time() - t0)
        return x, info.__dict__

    converged = False
    last_iter = 0

    for it in range(1, maxiter + 1):
        x = gauss_seidel_sweep_with_scipy(DL, U, b, x)

        if it % check_every == 0 or it == maxiter:
            r = b - A @ x
            abs_res = np.linalg.norm(r)
            rel_res = abs_res / b_norm
            history.append(rel_res)

            if verbose:
                print(f"GS iter={it}, rel_res={rel_res:.6e}, abs_res={abs_res:.6e}")

            if abs_res <= atol or rel_res <= rtol:
                converged = True
                last_iter = it
                break

        last_iter = it

    info = GSInfo(
        converged=converged,
        num_iter=last_iter,
        rel_res=rel_res,
        abs_res=abs_res,
        b_norm=b_norm,
        history=history,
        elapsed_time=time.time() - t0,
    )

    return x, info.__dict__


def residual_norm(A: Any, x: Any, b: Any) -> tuple[float, float]:
    A = as_csr_matrix(A)
    x = to_numpy_vector(x)
    b = to_numpy_vector(b)

    r = b - A @ x
    abs_res = np.linalg.norm(r)
    b_norm = np.linalg.norm(b)
    if b_norm == 0.0:
        b_norm = 1.0

    return abs_res, abs_res / b_norm
