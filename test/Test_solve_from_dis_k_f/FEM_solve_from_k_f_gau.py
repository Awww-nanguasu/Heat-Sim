from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike

from fealpy.mesh import UniformMesh2d
from fealpy.functionspace.lagrange_fe_space import LagrangeFESpace
from fealpy.decorator import cartesian
from fealpy.fem import BilinearForm, ScalarDiffusionIntegrator
from fealpy.fem import LinearForm, ScalarSourceIntegrator
from fealpy.fem import DirichletBC
from fealpy.solver import cg

import numpy as np
import torch

data = np.load(r"C:\git-workplace\HeatSim\ExperimentI\kf_quad.npz")

sample_id = 0

points = data["points"]
weights = data["quad_weights"]

k = data["k"][sample_id]
f = data["f"][sample_id]

print("points:", points.shape)
print("weights:", weights.shape)
print("k:", k.shape)
print("f:", f.shape)

device = 'cpu'

bm.set_backend('pytorch')
bm.set_default_device(device)
dtype = bm.float32

k = bm.tensor(k, dtype=dtype, device=device)
f = bm.tensor(f, dtype=dtype, device=device)

class Exp1():
    def __init__(self, dtype = bm.float32):
        self.domain = [0, 1, 0, 1]
        self.dtype = dtype
    
    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition"""
        x = p[..., 0]
        return bm.zeros(x.shape, dtype=self.dtype)
    
PDE = Exp1()

domain = PDE.domain
nx, ny = 100, 100

hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny

mesh = UniformMesh2d((0, nx, 0, ny), h=(hx, hy), origin=(domain[0], domain[2]), ftype=bm.float32)

space= LagrangeFESpace(mesh, p=1)
uh = space.function()
bform = BilinearForm(space)
DI = ScalarDiffusionIntegrator(k)
bform.add_integrator(DI)

lform = LinearForm(space)
SI = ScalarSourceIntegrator(f)
lform.add_integrator(SI)

A = bform.assembly()
F = lform.assembly()

A, F = DirichletBC(space, gd=PDE.dirichlet).apply(A, F)

uh[:] = cg(A, F, maxit=5000, atol=1e-14, rtol=1e-14)

import matplotlib.pyplot as plt

uh_2d = uh.reshape(nx + 1, ny + 1)

plt.figure(figsize=(6, 5))
plt.imshow(
    uh_2d.T,              # 转置一下，让 x/y 方向显示更自然
    origin="lower",
    extent=[0, 1, 0, 1],
    aspect="equal"
)
plt.colorbar(label="u_h")
plt.xlabel("x")
plt.ylabel("y")
plt.title("FEM solution $u_h$")
plt.show()