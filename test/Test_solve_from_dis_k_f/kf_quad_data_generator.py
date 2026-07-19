"""
generate_gp_kf_from_fealpy_quadrature_noflat.py

Generate 2D GP samples k and f directly on FEALPy cell quadrature points.

This version saves k and f in the original quadrature-point structure.
It does NOT save k_flat / f_flat.

Output .npz contains:
    points:        original ps shape, usually [..., 2]
    quad_weights: quadrature weights from FEALPy
    k:             [n_samples, *points.shape[:-1]]
    f:             [n_samples, *points.shape[:-1]]

Example:
    if ps.shape == (NC, NQ, 2), then k.shape == (n_samples, NC, NQ)
    if ps.shape == (NQ, NC, 2), then k.shape == (n_samples, NQ, NC)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu") and hasattr(x.cpu(), "numpy"):
        return x.cpu().numpy()
    return np.asarray(x)


def generate_fealpy_quadrature_points(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    from fealpy.backend import backend_manager as bm
    from fealpy.mesh import UniformMesh2d

    backend = cfg["runtime"].get("backend", "numpy")
    device = cfg["runtime"].get("device", "cpu")

    bm.set_backend(backend)
    if backend == "pytorch":
        bm.set_default_device(device)

    domain = cfg["mesh"]["domain"]
    nx = int(cfg["mesh"]["nx"])
    ny = int(cfg["mesh"]["ny"])

    hx = (domain[1] - domain[0]) / nx
    hy = (domain[3] - domain[2]) / ny

    mesh = UniformMesh2d((0, nx, 0, ny), h=(hx, hy), origin=(domain[0], domain[2]))

    order = int(cfg["quadrature"]["order"])
    entity = cfg["quadrature"].get("entity", "cell")

    cqf = mesh.quadrature_formula(order, entity)
    bcs, ws = cqf.get_quadrature_points_and_weights()
    ps = mesh.bc_to_point(bcs)

    points = to_numpy(ps).astype(np.float64)
    quad_weights = to_numpy(ws).astype(np.float64)

    if points.shape[-1] != 2:
        raise ValueError(f"Expected last dimension of points to be 2, got {points.shape}")

    return points, quad_weights


def sample_gp_rff(
    points_flat: np.ndarray,
    n_samples: int,
    mean: float,
    std: float,
    length_scale: float,
    rng: np.random.Generator,
    n_features: int = 4096,
    chunk_size: int = 50000,
) -> np.ndarray:
    n_points, dim = points_flat.shape
    omega = rng.normal(loc=0.0, scale=1.0 / length_scale, size=(n_features, dim))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(n_features,))
    coeff = rng.standard_normal(size=(n_features, n_samples))

    out = np.empty((n_samples, n_points), dtype=np.float64)
    scale = np.sqrt(2.0 / n_features)

    for start in range(0, n_points, chunk_size):
        end = min(start + chunk_size, n_points)
        pts = points_flat[start:end]
        phi = scale * np.cos(pts @ omega.T + phase[None, :])
        values = mean + std * (phi @ coeff)
        out[:, start:end] = values.T

    return out


def rbf_covariance(points_flat: np.ndarray, std: float, length_scale: float) -> np.ndarray:
    diff = points_flat[:, None, :] - points_flat[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    return (std ** 2) * np.exp(-0.5 * dist2 / (length_scale ** 2))


def sample_gp_exact(
    points_flat: np.ndarray,
    n_samples: int,
    mean: float,
    std: float,
    length_scale: float,
    rng: np.random.Generator,
    jitter: float = 1e-8,
) -> np.ndarray:
    K = rbf_covariance(points_flat, std=std, length_scale=length_scale)
    K = K + jitter * np.eye(K.shape[0])
    L = np.linalg.cholesky(K)
    z = rng.standard_normal(size=(points_flat.shape[0], n_samples))
    return mean + (L @ z).T


def sample_field(
    points: np.ndarray,
    n_samples: int,
    field_cfg: dict[str, Any],
    seed: int,
    method: str,
    rff_cfg: dict[str, Any],
    exact_cfg: dict[str, Any],
) -> np.ndarray:
    """Return values with shape [n_samples, *points.shape[:-1]]."""
    rng = np.random.default_rng(seed)

    mean = float(field_cfg["mean"])
    std = float(field_cfg["std"])
    length_scale = float(field_cfg["length_scale"])

    original_value_shape = points.shape[:-1]
    points_flat = points.reshape(-1, 2)

    if method == "rff":
        values_flat = sample_gp_rff(
            points_flat=points_flat,
            n_samples=n_samples,
            mean=mean,
            std=std,
            length_scale=length_scale,
            rng=rng,
            n_features=int(rff_cfg.get("n_features", 4096)),
            chunk_size=int(rff_cfg.get("chunk_size", 50000)),
        )
    elif method == "exact":
        values_flat = sample_gp_exact(
            points_flat=points_flat,
            n_samples=n_samples,
            mean=mean,
            std=std,
            length_scale=length_scale,
            rng=rng,
            jitter=float(exact_cfg.get("jitter", 1e-8)),
        )
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    return values_flat.reshape((n_samples, *original_value_shape))


def save_preview_plot(points: np.ndarray, k: np.ndarray, f: np.ndarray, path: str | Path, max_points: int = 200000) -> None:
    import matplotlib.pyplot as plt

    points_flat = points.reshape(-1, 2)
    k0 = k[0].reshape(-1)
    f0 = f[0].reshape(-1)

    if points_flat.shape[0] > max_points:
        idx = np.linspace(0, points_flat.shape[0] - 1, max_points).astype(int)
        points_flat = points_flat[idx]
        k0 = k0[idx]
        f0 = f0[idx]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    s0 = axes[0].scatter(points_flat[:, 0], points_flat[:, 1], c=k0, s=2)
    axes[0].set_title("k at quadrature points")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(s0, ax=axes[0])

    s1 = axes[1].scatter(points_flat[:, 0], points_flat[:, 1], c=f0, s=2)
    axes[1].set_title("f at quadrature points")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(s1, ax=axes[1])

    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="Data_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_path = Path(cfg["output"]["path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    points, quad_weights = generate_fealpy_quadrature_points(cfg)
    n_points = int(np.prod(points.shape[:-1]))

    n_samples = int(cfg["sampling"]["n_samples"])
    seed = int(cfg["sampling"]["seed"])
    method = cfg["sampling"]["method"]

    if method == "exact" and n_points > int(cfg["sampling"].get("exact_warning_threshold", 8000)):
        print(
            f"Warning: exact GP with {n_points} points will build a huge covariance matrix. "
            "Use method: rff for FEM quadrature datasets."
        )

    k = sample_field(
        points=points,
        n_samples=n_samples,
        field_cfg=cfg["fields"]["k"],
        seed=seed,
        method=method,
        rff_cfg=cfg.get("rff", {}),
        exact_cfg=cfg.get("exact", {}),
    )

    f = sample_field(
        points=points,
        n_samples=n_samples,
        field_cfg=cfg["fields"]["f"],
        seed=seed + int(cfg["sampling"].get("f_seed_offset", 12345)),
        method=method,
        rff_cfg=cfg.get("rff", {}),
        exact_cfg=cfg.get("exact", {}),
    )

    if bool(cfg["fields"]["k"].get("clip_positive", False)):
        min_value = float(cfg["fields"]["k"].get("min_value", 1e-6))
        k = np.maximum(k, min_value)

    np.savez_compressed(
        out_path,
        points=points,
        coords=points,
        quad_weights=quad_weights,
        k=k,
        f=f,
        domain=np.asarray(cfg["mesh"]["domain"], dtype=np.float64),
        nx=int(cfg["mesh"]["nx"]),
        ny=int(cfg["mesh"]["ny"]),
        quadrature_order=int(cfg["quadrature"]["order"]),
        method=method,
        n_samples=n_samples,
        seed=seed,
        k_mean=float(cfg["fields"]["k"]["mean"]),
        k_std=float(cfg["fields"]["k"]["std"]),
        k_length_scale=float(cfg["fields"]["k"]["length_scale"]),
        f_mean=float(cfg["fields"]["f"]["mean"]),
        f_std=float(cfg["fields"]["f"]["std"]),
        f_length_scale=float(cfg["fields"]["f"]["length_scale"]),
    )

    preview_path = cfg["output"].get("preview_plot", None)
    if preview_path:
        save_preview_plot(points, k, f, preview_path)

    print("Generated FEALPy quadrature GP data without flat arrays")
    print(f"points shape: {points.shape}")
    print(f"quad_weights shape: {quad_weights.shape}")
    print(f"k shape: {k.shape}, mean={k.mean():.6f}, std={k.std():.6f}")
    print(f"f shape: {f.shape}, mean={f.mean():.6f}, std={f.std():.6f}")
    print(f"saved to: {out_path}")


if __name__ == "__main__":
    main()
