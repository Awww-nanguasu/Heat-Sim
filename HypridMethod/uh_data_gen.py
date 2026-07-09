
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.typing import TensorLike
from fealpy.mesh import UniformMesh2d
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import (
    BilinearForm,
    LinearForm,
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    DirichletBC,
)
from fealpy.solver import cg


class Exp1:
    def __init__(self, dtype=bm.float32):
        self.domain = [0, 1, 0, 1]
        self.dtype = dtype

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        x = p[..., 0]
        return bm.zeros(x.shape, dtype=self.dtype)


def to_numpy(x):
    """Convert FEALPy/PyTorch backend tensor to numpy."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu") and hasattr(x.cpu(), "numpy"):
        return x.cpu().numpy()
    return np.asarray(x)


def cg_solve(A, F, *, maxit: int, atol: float, rtol: float):
    """Handle different FEALPy cg return styles."""
    out = cg(A, F, maxit=maxit, atol=atol, rtol=rtol)
    if isinstance(out, tuple):
        return out[0], out[1:]
    return out, None


def relative_residual(A, u, F) -> float:
    r = F - A @ u
    rn = bm.linalg.norm(r)
    fn = bm.linalg.norm(F)
    rn = float(to_numpy(rn))
    fn = float(to_numpy(fn))
    if fn == 0.0:
        fn = 1.0
    return rn / fn


def main():
    parser = argparse.ArgumentParser(description="Batch solve generated 2D Poisson equations with FEALPy.")
    parser.add_argument("--input", type=str, default=r"C:\git-workplace\HeatSim\HypridMethod\Data\data\mionet_gp_2d_5000")
    parser.add_argument("--output-dir", type=str, default=r"C:\git-workplace\HeatSim\HypridMethod\Data\data\mionet_gp_2d_5000\solved_5000")
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--maxit", type=int, default=5000)
    parser.add_argument("--atol", type=float, default=1e-14)
    parser.add_argument("--rtol", type=float, default=1e-14)
    parser.add_argument("--backend", type=str, default="pytorch", choices=["numpy", "pytorch"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--check-residual", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bm.set_backend(args.backend)
    if args.backend == "pytorch":
        bm.set_default_device(args.device)

    if args.dtype == "float64":
        torch.set_default_dtype(torch.float64)
        dtype = bm.float64
        np_dtype = np.float64
        ftype = bm.float64
    else:
        torch.set_default_dtype(torch.float32)
        dtype = bm.float32
        np_dtype = np.float32
        ftype = bm.float32

    input_dir = Path(args.input)

    k_all = np.load(input_dir / "k_quad.npy", mmap_mode="r")
    f_all = np.load(input_dir / "f_quad.npy", mmap_mode="r")

    quad_points = np.load(input_dir / "quad_points.npy", mmap_mode="r")
    quad_weights = np.load(input_dir / "quad_weights.npy", mmap_mode="r")
    
    n_samples = k_all.shape[0]
    start = int(args.start)
    end = n_samples if args.end is None else int(args.end)
    end = min(end, n_samples)

    if not (0 <= start < end <= n_samples):
        raise ValueError(f"Invalid sample range: start={start}, end={end}, n_samples={n_samples}")

    print("Input:", input_path)
    print("k_all shape:", k_all.shape)
    print("f_all shape:", f_all.shape)

    print("Solving samples:", start, "to", end - 1)

    PDE = Exp1(dtype=dtype)
    domain = PDE.domain

    nx, ny = args.nx, args.ny
    hx = (domain[1] - domain[0]) / nx
    hy = (domain[3] - domain[2]) / ny

    mesh = UniformMesh2d(
        (0, nx, 0, ny),
        h=(hx, hy),
        origin=(domain[0], domain[2]),
        ftype=ftype,
    )

    space = LagrangeFESpace(mesh, p=1)
    uh = space.function()

    ndof = len(uh[:])
    print("ndof:", ndof)

    u_path = output_dir / "u_train.npy"
    residual_path = output_dir / "relative_residual.npy"
    time_path = output_dir / "solve_time.npy"
    status_path = output_dir / "status.npy"

    resume = u_path.exists() and residual_path.exists() and time_path.exists() and status_path.exists() and not args.overwrite

    if resume:
        u_train = np.lib.format.open_memmap(u_path, mode="r+", dtype=np_dtype, shape=(n_samples, ndof))
        rel_res_arr = np.lib.format.open_memmap(residual_path, mode="r+", dtype=np.float64, shape=(n_samples,))
        solve_time_arr = np.lib.format.open_memmap(time_path, mode="r+", dtype=np.float64, shape=(n_samples,))
        status_arr = np.lib.format.open_memmap(status_path, mode="r+", dtype=np.int32, shape=(n_samples,))
        print("Resume mode: existing output arrays are opened.")
    else:
        u_train = np.lib.format.open_memmap(u_path, mode="w+", dtype=np_dtype, shape=(n_samples, ndof))
        rel_res_arr = np.lib.format.open_memmap(residual_path, mode="w+", dtype=np.float64, shape=(n_samples,))
        solve_time_arr = np.lib.format.open_memmap(time_path, mode="w+", dtype=np.float64, shape=(n_samples,))
        status_arr = np.lib.format.open_memmap(status_path, mode="w+", dtype=np.int32, shape=(n_samples,))
        rel_res_arr[:] = np.nan
        solve_time_arr[:] = np.nan
        status_arr[:] = 0

    total_t0 = time.time()

    for i in range(start, end):
        t0 = time.time()

        k_np = np.asarray(k_all[i], dtype=np_dtype)
        f_np = np.asarray(f_all[i], dtype=np_dtype)

        k = bm.tensor(k_np, dtype=dtype, device=args.device)
        f = bm.tensor(f_np, dtype=dtype, device=args.device)

        bform = BilinearForm(space)
        bform.add_integrator(ScalarDiffusionIntegrator(k))
        A = bform.assembly()

        lform = LinearForm(space)
        lform.add_integrator(ScalarSourceIntegrator(f))
        F = lform.assembly()

        A, F = DirichletBC(space, gd=PDE.dirichlet).apply(A, F)

        sol, cg_info = cg_solve(A, F, maxit=args.maxit, atol=args.atol, rtol=args.rtol)
        uh[:] = sol

        u_train[i, :] = to_numpy(uh[:]).astype(np_dtype)

        if args.check_residual:
            rel_res = relative_residual(A, uh[:], F)
        else:
            rel_res = np.nan

        dt = time.time() - t0
        rel_res_arr[i] = rel_res
        solve_time_arr[i] = dt
        status_arr[i] = 1

        if hasattr(u_train, "flush"):
            u_train.flush()
            rel_res_arr.flush()
            solve_time_arr.flush()
            status_arr.flush()

        if (i - start) % args.log_every == 0 or i == end - 1:
            if args.check_residual:
                print(f"[{i + 1}/{n_samples}] time={dt:.3f}s, rel_res={rel_res:.3e}")
            else:
                print(f"[{i + 1}/{n_samples}] time={dt:.3f}s")

    np.savez(
        output_dir / "solve_info.npz",
        input=str(input_path),
        n_samples=n_samples,
        solved_start=start,
        solved_end=end,
        nx=nx,
        ny=ny,
        ndof=ndof,
        dtype=args.dtype,
        maxit=args.maxit,
        atol=args.atol,
        rtol=args.rtol,
        total_time=time.time() - total_t0,
        u_train_path=str(u_path),
        relative_residual_path=str(residual_path),
        solve_time_path=str(time_path),
        status_path=str(status_path),
    )

    print("Done.")
    print("u_train saved to:", u_path)
    print("u_train shape:", u_train.shape)
    print("solve_info saved to:", output_dir / "solve_info.npz")


if __name__ == "__main__":
    main()
