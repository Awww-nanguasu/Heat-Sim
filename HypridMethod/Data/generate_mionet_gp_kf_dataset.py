"""
generate_mionet_gp_kf_dataset.py

Generate 2D paper-like GP samples for MIONet + FEM reproduction.

This script generates the SAME random field sample evaluated on two point sets:

1. sensor_points:
       Used as MIONet branch input.
       Example: 100 x 100 = 10000 points.

2. quad_points:
       FEALPy cell quadrature points.
       Used for FEM assembly of A(k) and F(f).

For each sample i:
       k_sensor[i] and k_quad[i] come from the same k_i(x, y)
       f_sensor[i] and f_quad[i] come from the same f_i(x, y)

This is important. Do NOT generate sensor values and quadrature values with
two independent GP draws.

Recommended output for 5000 samples:
       output.format = npy_dir

It creates:
       dataset_dir/
           sensor_points.npy     [n_sensor, 2]
           quad_points.npy       original FEALPy quadrature shape [..., 2]
           quad_weights.npy
           output_points.npy     optional output/query coordinates
           k_sensor.npy          [n_samples, n_sensor]
           f_sensor.npy          [n_samples, n_sensor]
           k_quad.npy            [n_samples, *quad_points.shape[:-1]]
           f_quad.npy            [n_samples, *quad_points.shape[:-1]]
           metadata.yaml

For small debugging datasets, you can use:
       output.format = npz

GP settings follow the paper-like 2D experiment:
       k ~ GP(mean=1.0, std=0.2, RBF length_scale=0.2)
       f ~ GP(mean=0.0, std=1.0, RBF length_scale=0.2)

Recommended:
       method = rff
because exact GP sampling is too expensive for 100x100 sensors plus FEM
quadrature points.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def to_numpy(x: Any) -> np.ndarray:
    """Convert FEALPy backend tensor / torch tensor / numpy array to numpy."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu") and hasattr(x.cpu(), "numpy"):
        return x.cpu().numpy()
    return np.asarray(x)


def make_regular_grid_points(
    domain: list[float],
    nx: int,
    ny: int,
    include_boundary: bool = True,
) -> np.ndarray:
    """Create a regular grid on [x0,x1] x [y0,y1]."""
    x0, x1, y0, y1 = map(float, domain)

    if include_boundary:
        xs = np.linspace(x0, x1, nx)
        ys = np.linspace(y0, y1, ny)
    else:
        hx = (x1 - x0) / nx
        hy = (y1 - y0) / ny
        xs = x0 + (np.arange(nx) + 0.5) * hx
        ys = y0 + (np.arange(ny) + 0.5) * hy

    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([X.reshape(-1), Y.reshape(-1)], axis=-1)


def generate_sensor_points(cfg: dict[str, Any]) -> np.ndarray:
    domain = cfg["mesh"]["domain"]
    sx = int(cfg["sensor"]["nx"])
    sy = int(cfg["sensor"]["ny"])
    include_boundary = bool(cfg["sensor"].get("include_boundary", True))
    return make_regular_grid_points(domain, sx, sy, include_boundary).astype(np.float64)


def generate_output_points(cfg: dict[str, Any], sensor_points: np.ndarray) -> np.ndarray:
    """Generate optional output/query coordinates for MIONet trunk.

    kind:
        mesh_nodes: regular FEM-like nodes, shape [(nx+1)*(ny+1), 2]
        sensor: use sensor_points
        none: return empty array
    """
    kind = cfg.get("output_points", {}).get("kind", "mesh_nodes")
    domain = cfg["mesh"]["domain"]

    if kind == "none":
        return np.empty((0, 2), dtype=np.float64)
    if kind == "sensor":
        return sensor_points.copy()
    if kind == "mesh_nodes":
        nx = int(cfg["mesh"]["nx"])
        ny = int(cfg["mesh"]["ny"])
        return make_regular_grid_points(domain, nx + 1, ny + 1, True).astype(np.float64)
    raise ValueError(f"Unknown output_points.kind: {kind}")


def generate_fealpy_quadrature_points(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Generate FEALPy cell quadrature points."""
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

    order = int(cfg["quadrature"].get("order", 4))
    entity = cfg["quadrature"].get("entity", "cell")

    cqf = mesh.quadrature_formula(order, entity)
    bcs, ws = cqf.get_quadrature_points_and_weights()
    ps = mesh.bc_to_point(bcs)

    quad_points = to_numpy(ps).astype(np.float64)
    quad_weights = to_numpy(ws).astype(np.float64)

    if quad_points.shape[-1] != 2:
        raise ValueError(f"Expected quad_points.shape[-1] == 2, got {quad_points.shape}")

    return quad_points, quad_weights


def open_output_array(
    output_format: str,
    base_dir: Path,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    if output_format == "npy_dir":
        path = base_dir / f"{name}.npy"
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    if output_format == "npz":
        return np.empty(shape, dtype=dtype)
    raise ValueError(f"Unknown output.format: {output_format}")


def rff_fill_arrays(
    *,
    point_sets: dict[str, np.ndarray],
    out_arrays: dict[str, np.ndarray],
    n_samples: int,
    mean: float,
    std: float,
    length_scale: float,
    seed: int,
    n_features: int,
    point_chunk_size: int,
    sample_chunk_size: int,
    dtype: np.dtype,
) -> None:
    """Fill output arrays with one RFF GP field evaluated on multiple point sets.

    This guarantees all point sets use the same GP sample: same omega, phase,
    and sample coefficients.
    """
    rng = np.random.default_rng(seed)
    dtype = np.dtype(dtype)

    omega = rng.normal(0.0, 1.0 / length_scale, size=(n_features, 2)).astype(dtype)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(n_features,)).astype(dtype)

    scale = np.asarray(np.sqrt(2.0 / n_features), dtype=dtype)
    mean = np.asarray(mean, dtype=dtype)
    std = np.asarray(std, dtype=dtype)

    flat_out = {name: arr.reshape((n_samples, -1)) for name, arr in out_arrays.items()}

    for sample_start in range(0, n_samples, sample_chunk_size):
        sample_end = min(sample_start + sample_chunk_size, n_samples)
        bs = sample_end - sample_start
        coeff = rng.standard_normal(size=(n_features, bs)).astype(dtype)

        for name, points in point_sets.items():
            pts_all = np.asarray(points, dtype=dtype)
            out = flat_out[name]
            n_points = pts_all.shape[0]

            for point_start in range(0, n_points, point_chunk_size):
                point_end = min(point_start + point_chunk_size, n_points)
                pts = pts_all[point_start:point_end]
                phi = scale * np.cos(pts @ omega.T + phase[None, :])
                values = mean + std * (phi @ coeff)
                out[sample_start:sample_end, point_start:point_end] = values.T

        print(
            f"  samples {sample_start:>6d}:{sample_end:<6d} done (seed={seed})",
            flush=True,
        )


def rbf_covariance(points: np.ndarray, std: float, length_scale: float) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    return (std**2) * np.exp(-0.5 * dist2 / (length_scale**2))


def exact_fill_arrays(
    *,
    point_sets: dict[str, np.ndarray],
    out_arrays: dict[str, np.ndarray],
    n_samples: int,
    mean: float,
    std: float,
    length_scale: float,
    seed: int,
    jitter: float,
    sample_chunk_size: int,
    dtype: np.dtype,
) -> None:
    """Exact GP sampling over the concatenation of all point sets.

    Only use for small debugging cases.
    """
    rng = np.random.default_rng(seed)
    names = list(point_sets.keys())
    sizes = [point_sets[name].shape[0] for name in names]
    starts = np.cumsum([0] + sizes[:-1])
    ends = np.cumsum(sizes)

    all_points = np.concatenate([point_sets[name] for name in names], axis=0)
    K = rbf_covariance(all_points, std=std, length_scale=length_scale)
    K = K + jitter * np.eye(K.shape[0])
    L = np.linalg.cholesky(K)

    flat_out = {name: arr.reshape((n_samples, -1)) for name, arr in out_arrays.items()}

    for sample_start in range(0, n_samples, sample_chunk_size):
        sample_end = min(sample_start + sample_chunk_size, n_samples)
        bs = sample_end - sample_start
        z = rng.standard_normal(size=(all_points.shape[0], bs))
        values_all = mean + (L @ z).T

        for name, s, e in zip(names, starts, ends):
            flat_out[name][sample_start:sample_end, :] = values_all[:, s:e].astype(dtype)

        print(
            f"  samples {sample_start:>6d}:{sample_end:<6d} done (exact seed={seed})",
            flush=True,
        )


def fill_field(
    *,
    field_name: str,
    cfg: dict[str, Any],
    point_sets: dict[str, np.ndarray],
    out_arrays: dict[str, np.ndarray],
    dtype: np.dtype,
) -> None:
    sampling_cfg = cfg["sampling"]
    method = sampling_cfg.get("method", "rff")

    field_cfg = cfg["fields"][field_name]
    mean = float(field_cfg["mean"])
    std = float(field_cfg["std"])
    length_scale = float(field_cfg["length_scale"])

    base_seed = int(sampling_cfg.get("seed", 0))
    if field_name == "k":
        seed = base_seed
    elif field_name == "f":
        seed = base_seed + int(sampling_cfg.get("f_seed_offset", 12345))
    else:
        raise ValueError(field_name)

    n_samples = int(sampling_cfg["n_samples"])
    sample_chunk_size = int(sampling_cfg.get("sample_chunk_size", 128))

    print(f"Generating {field_name}: method={method}, n_samples={n_samples}")

    if method == "rff":
        rff_cfg = cfg.get("rff", {})
        rff_fill_arrays(
            point_sets=point_sets,
            out_arrays=out_arrays,
            n_samples=n_samples,
            mean=mean,
            std=std,
            length_scale=length_scale,
            seed=seed,
            n_features=int(rff_cfg.get("n_features", 4096)),
            point_chunk_size=int(rff_cfg.get("point_chunk_size", 2048)),
            sample_chunk_size=sample_chunk_size,
            dtype=dtype,
        )
    elif method == "exact":
        total_points = sum(points.shape[0] for points in point_sets.values())
        threshold = int(sampling_cfg.get("exact_warning_threshold", 8000))
        if total_points > threshold:
            print(
                f"Warning: exact GP over {total_points} points is huge. Use method: rff."
            )
        exact_cfg = cfg.get("exact", {})
        exact_fill_arrays(
            point_sets=point_sets,
            out_arrays=out_arrays,
            n_samples=n_samples,
            mean=mean,
            std=std,
            length_scale=length_scale,
            seed=seed,
            jitter=float(exact_cfg.get("jitter", 1e-8)),
            sample_chunk_size=sample_chunk_size,
            dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    if field_name == "k" and bool(field_cfg.get("clip_positive", False)):
        min_value = float(field_cfg.get("min_value", 1e-6))
        for arr in out_arrays.values():
            np.maximum(arr, min_value, out=arr)


def save_preview_plot(
    *,
    sensor_points: np.ndarray,
    k_sensor: np.ndarray,
    f_sensor: np.ndarray,
    path: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    k0 = np.asarray(k_sensor[0]).reshape(-1)
    f0 = np.asarray(f_sensor[0]).reshape(-1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)

    s0 = axes[0].scatter(sensor_points[:, 0], sensor_points[:, 1], c=k0, s=3)
    axes[0].set_title("k on sensor points")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(s0, ax=axes[0])

    s1 = axes[1].scatter(sensor_points[:, 0], sensor_points[:, 1], c=f0, s=3)
    axes[1].set_title("f on sensor points")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(s1, ax=axes[1])

    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="mionet_gp_dataset.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dtype = np.dtype(cfg["output"].get("dtype", "float32"))
    output_format = cfg["output"].get("format", "npy_dir")
    n_samples = int(cfg["sampling"]["n_samples"])

    sensor_points = generate_sensor_points(cfg)
    quad_points, quad_weights = generate_fealpy_quadrature_points(cfg)
    output_points = generate_output_points(cfg, sensor_points)

    sensor_flat = sensor_points.reshape(-1, 2)
    quad_flat = quad_points.reshape(-1, 2)
    n_sensor = sensor_flat.shape[0]
    quad_value_shape = quad_points.shape[:-1]

    print("Point sets:")
    print(f"  sensor_points shape: {sensor_points.shape}")
    print(f"  quad_points shape:   {quad_points.shape}")
    print(f"  quad_weights shape:  {quad_weights.shape}")
    print(f"  output_points shape: {output_points.shape}")
    print(f"  n_samples:           {n_samples}")
    print(f"  dtype:               {dtype}")
    print(f"  output format:       {output_format}")

    output_path = Path(cfg["output"]["path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "npy_dir":
        dataset_dir = output_path
        if dataset_dir.exists() and bool(cfg["output"].get("overwrite", False)):
            shutil.rmtree(dataset_dir)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        np.save(dataset_dir / "sensor_points.npy", sensor_points.astype(dtype))
        np.save(dataset_dir / "quad_points.npy", quad_points.astype(dtype))
        np.save(dataset_dir / "quad_weights.npy", quad_weights.astype(dtype))
        np.save(dataset_dir / "output_points.npy", output_points.astype(dtype))

        k_sensor = open_output_array(output_format, dataset_dir, "k_sensor", (n_samples, n_sensor), dtype)
        f_sensor = open_output_array(output_format, dataset_dir, "f_sensor", (n_samples, n_sensor), dtype)
        k_quad = open_output_array(output_format, dataset_dir, "k_quad", (n_samples, *quad_value_shape), dtype)
        f_quad = open_output_array(output_format, dataset_dir, "f_quad", (n_samples, *quad_value_shape), dtype)

    elif output_format == "npz":
        dataset_dir = output_path.parent
        k_sensor = open_output_array(output_format, dataset_dir, "k_sensor", (n_samples, n_sensor), dtype)
        f_sensor = open_output_array(output_format, dataset_dir, "f_sensor", (n_samples, n_sensor), dtype)
        k_quad = open_output_array(output_format, dataset_dir, "k_quad", (n_samples, *quad_value_shape), dtype)
        f_quad = open_output_array(output_format, dataset_dir, "f_quad", (n_samples, *quad_value_shape), dtype)
    else:
        raise ValueError(f"Unknown output.format: {output_format}")

    point_sets = {"sensor": sensor_flat, "quad": quad_flat}

    fill_field(
        field_name="k",
        cfg=cfg,
        point_sets=point_sets,
        out_arrays={"sensor": k_sensor, "quad": k_quad},
        dtype=dtype,
    )

    fill_field(
        field_name="f",
        cfg=cfg,
        point_sets=point_sets,
        out_arrays={"sensor": f_sensor, "quad": f_quad},
        dtype=dtype,
    )

    metadata = {
        "description": "MIONet/FEM GP dataset. Sensor and quadrature values come from the same GP samples.",
        "n_samples": n_samples,
        "dtype": str(dtype),
        "sensor_points_shape": list(sensor_points.shape),
        "quad_points_shape": list(quad_points.shape),
        "quad_weights_shape": list(quad_weights.shape),
        "output_points_shape": list(output_points.shape),
        "k_sensor_shape": list(k_sensor.shape),
        "f_sensor_shape": list(f_sensor.shape),
        "k_quad_shape": list(k_quad.shape),
        "f_quad_shape": list(f_quad.shape),
        "mesh": cfg["mesh"],
        "sensor": cfg["sensor"],
        "quadrature": cfg["quadrature"],
        "sampling": cfg["sampling"],
        "fields": cfg["fields"],
        "rff": cfg.get("rff", {}),
        "exact": cfg.get("exact", {}),
        "output_format": output_format,
    }

    if output_format == "npy_dir":
        for arr in [k_sensor, f_sensor, k_quad, f_quad]:
            if hasattr(arr, "flush"):
                arr.flush()
        save_yaml(metadata, output_path / "metadata.yaml")
    else:
        np.savez_compressed(
            output_path,
            sensor_points=sensor_points.astype(dtype),
            quad_points=quad_points.astype(dtype),
            quad_weights=quad_weights.astype(dtype),
            output_points=output_points.astype(dtype),
            k_sensor=np.asarray(k_sensor),
            f_sensor=np.asarray(f_sensor),
            k_quad=np.asarray(k_quad),
            f_quad=np.asarray(f_quad),
            metadata_json=json.dumps(metadata),
        )

    preview_path = cfg["output"].get("preview_plot", None)
    if preview_path:
        save_preview_plot(
            sensor_points=sensor_points,
            k_sensor=np.asarray(k_sensor),
            f_sensor=np.asarray(f_sensor),
            path=preview_path,
        )

    print("Done.")
    print(f"Saved to: {output_path}")
    print(f"k_sensor shape: {k_sensor.shape}")
    print(f"f_sensor shape: {f_sensor.shape}")
    print(f"k_quad shape:   {k_quad.shape}")
    print(f"f_quad shape:   {f_quad.shape}")


if __name__ == "__main__":
    main()
