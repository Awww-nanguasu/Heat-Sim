"""
mionet_linear.py

A PyTorch implementation of MIONet for PDE solution/residual-correction operators,
designed to preserve strict linearity with respect to the source/residual input.

Typical target operator:
    M(k, f)(y) -> u(y)

For heat conduction / elliptic PDEs, k can be thermal conductivity or material
parameters, f can be source term or residual function, and y are query coordinates.

Key property:
    For fixed k and y,
    M(k, a*f1 + b*f2)(y) = a*M(k, f1)(y) + b*M(k, f2)(y)

This is enforced by:
    1. a nonlinear branch for k
    2. a strictly linear branch for f/residual: nn.Linear(..., bias=False)
    3. no additive output bias
    4. multiplicative merge with k-branch and trunk features
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MIONetConfig:
    """Configuration for LinearMIONet.

    Args:
        k_dim: Dimension of the coefficient/material input encoding.
        f_dim: Dimension of the source/residual input encoding.
        coord_dim: Spatial coordinate dimension, e.g. 1 for x, 2 for (x, y).
        latent_dim: Shared feature dimension p in the MIONet representation.
        k_hidden: Hidden layer widths for the nonlinear k branch.
        trunk_hidden: Hidden layer widths for the trunk network.
        activation: Activation module class used in nonlinear branches.
    """

    k_dim: int
    f_dim: int
    coord_dim: int
    latent_dim: int = 128
    k_hidden: Sequence[int] = (128, 128)
    trunk_hidden: Sequence[int] = (128, 128)
    activation: type[nn.Module] = nn.Tanh


def make_mlp(
    in_dim: int,
    out_dim: int,
    hidden: Sequence[int],
    activation: type[nn.Module] = nn.Tanh,
    *,
    final_bias: bool = True,
) -> nn.Sequential:
    """Build a standard MLP."""
    layers: list[nn.Module] = []
    prev = in_dim
    for width in hidden:
        layers.append(nn.Linear(prev, width))
        layers.append(activation())
        prev = width
    layers.append(nn.Linear(prev, out_dim, bias=final_bias))
    return nn.Sequential(*layers)


class StrictLinearBranch(nn.Module):
    """A strictly linear branch net.

    This module is exactly a linear map f -> Wf.

    It intentionally has:
        - no bias
        - no activation
        - no normalization
        - no dropout

    Those omissions are what preserve exact linearity.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class LinearMIONet(nn.Module):
    """MIONet with strict linearity in the second input.

    The model computes

        M(k, f)(y) = sum_j branch_k(k)_j * branch_f(f)_j * trunk(y)_j

    where branch_f is a strictly linear map.

    Shapes:
        k:      [batch, k_dim]
        f:      [batch, f_dim]
        coords: [n_points, coord_dim] or [batch, n_points, coord_dim]

    Returns:
        u:      [batch, n_points]
    """

    def __init__(self, cfg: MIONetConfig):
        super().__init__()
        self.cfg = cfg

        # Nonlinear branch: coefficient/material field k.
        self.k_branch = make_mlp(
            cfg.k_dim,
            cfg.latent_dim,
            cfg.k_hidden,
            cfg.activation,
            final_bias=True,
        )

        # Strictly linear branch: source term / residual function f or r.
        self.f_branch = StrictLinearBranch(cfg.f_dim, cfg.latent_dim)

        # Trunk net: coordinate-to-basis representation.
        self.trunk = make_mlp(
            cfg.coord_dim,
            cfg.latent_dim,
            cfg.trunk_hidden,
            cfg.activation,
            final_bias=True,
        )

        # Important:
        # Do NOT add a global scalar bias here.
        # A bias would violate M(k, 0) = 0 and linearity in f.

    def forward(self, k: Tensor, f: Tensor, coords: Tensor) -> Tensor:
        """Evaluate M(k, f)(coords)."""
        if k.ndim != 2:
            raise ValueError(f"k must have shape [batch, k_dim], got {tuple(k.shape)}")
        if f.ndim != 2:
            raise ValueError(f"f must have shape [batch, f_dim], got {tuple(f.shape)}")
        if k.shape[0] != f.shape[0]:
            raise ValueError("k and f must have the same batch size")

        batch_size = k.shape[0]

        k_feat = self.k_branch(k)        # [B, P]
        f_feat = self.f_branch(f)        # [B, P], strictly linear in f
        branch_feat = k_feat * f_feat    # [B, P]

        if coords.ndim == 2:
            # Shared query coordinates for the whole batch.
            # coords: [N, d]
            trunk_feat = self.trunk(coords)          # [N, P]
            out = branch_feat @ trunk_feat.T         # [B, N]
            return out

        if coords.ndim == 3:
            # Different query coordinates for each batch item.
            # coords: [B, N, d]
            if coords.shape[0] != batch_size:
                raise ValueError("coords batch dimension must match k/f batch size")
            trunk_feat = self.trunk(coords.reshape(-1, coords.shape[-1]))
            trunk_feat = trunk_feat.reshape(batch_size, coords.shape[1], -1)  # [B, N, P]
            out = torch.einsum("bp,bnp->bn", branch_feat, trunk_feat)
            return out

        raise ValueError(
            f"coords must have shape [n_points, coord_dim] or "
            f"[batch, n_points, coord_dim], got {tuple(coords.shape)}"
        )

    @torch.no_grad()
    def check_linearity_in_f(
        self,
        k: Tensor,
        f1: Tensor,
        f2: Tensor,
        coords: Tensor,
        a: float = 1.7,
        b: float = -0.4,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ) -> tuple[bool, float]:
        """Numerically verify strict linearity in f.

        Returns:
            (passed, max_abs_error)
        """
        lhs = self(k, a * f1 + b * f2, coords)
        rhs = a * self(k, f1, coords) + b * self(k, f2, coords)
        err = (lhs - rhs).abs().max().item()
        return torch.allclose(lhs, rhs, atol=atol, rtol=rtol), err


def mse_operator_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Basic supervised operator regression loss.

    pred and target are usually sampled solution values:
        [batch, n_points]
    """
    return torch.mean((pred - target) ** 2)


def demo_train_step() -> None:
    """Minimal example of one training step.

    This is only a shape/demo function. Replace random tensors with:
        k_samples: encoded thermal conductivity / PDE coefficients
        f_samples: encoded heat source or residual function
        coords: query points
        u_samples: FEM/FDM/FVM solution values at coords
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = MIONetConfig(
        k_dim=64,          # e.g. sampled thermal conductivity k(x)
        f_dim=64,          # e.g. sampled heat source f(x) or residual r(x)
        coord_dim=1,       # use 2 for 2D heat conduction
        latent_dim=128,
    )
    model = LinearMIONet(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 8
    n_points = 100

    k = torch.randn(batch_size, cfg.k_dim, device=device)
    f = torch.randn(batch_size, cfg.f_dim, device=device)
    coords = torch.linspace(0, 1, n_points, device=device).reshape(n_points, 1)
    u_target = torch.randn(batch_size, n_points, device=device)

    pred = model(k, f, coords)
    loss = mse_operator_loss(pred, u_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    f1 = torch.randn_like(f)
    f2 = torch.randn_like(f)
    passed, err = model.check_linearity_in_f(k, f1, f2, coords)
    print(f"loss={loss.item():.6f}, linearity_check={passed}, max_error={err:.3e}")


if __name__ == "__main__":
    demo_train_step()
