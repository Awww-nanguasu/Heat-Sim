#此测试文件用于测试，离散形式k，f， 与连续形式k，f的输入是否对有限元结果有影响。
from typing import Sequence
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike

from fealpy.mesh import UniformMesh2d
from fealpy.functionspace.lagrange_fe_space import LagrangeFESpace
from fealpy.decorator import cartesian
from fealpy.fem import BilinearForm, ScalarDiffusionIntegrator
from fealpy.fem import LinearForm, ScalarSourceIntegrator
from fealpy.fem import DirichletBC
from fealpy.solver import cg

device = 'cpu'
bm.set_backend('pytorch')
bm.set_default_device(device)


class Exp0002():
    def __init__(self):
        self.domain = [0, 1, 0, 1] 

    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Compute exact solution"""
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi
        return bm.sin(pi * x) * bm.sin(pi * y)

    @cartesian
    def diffusion_coef(self, p: TensorLike) -> TensorLike:
        """Variable diffusion coefficient k(x, y)"""
        x = p[..., 0]
        y = p[..., 1]
        return 1.0 + x + y

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """f = -div(k grad u), where k = 1 + x + y"""
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi

        u = bm.sin(pi * x) * bm.sin(pi * y)

        ux = pi * bm.cos(pi * x) * bm.sin(pi * y)
        uy = pi * bm.sin(pi * x) * bm.cos(pi * y)

        k = 1.0 + x + y

        # -div(k grad u)
        # = -kx * ux - ky * uy - k * laplace(u)
        # kx = 1, ky = 1, laplace(u) = -2*pi^2*u
        return 2 * pi**2 * k * u - ux - uy

    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Dirichlet boundary condition"""
        return self.solution(p)

PDE = Exp0002()

domain = PDE.domain
nx, ny = 100, 100

hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny

mesh = UniformMesh2d((0, nx, 0, ny), h=(hx, hy), origin=(domain[0], domain[2]))

cqf = mesh.quadrature_formula(4, 'cell')
bcs, ws = cqf.get_quadrature_points_and_weights()
ps = mesh.bc_to_point(bcs)

space= LagrangeFESpace(mesh, p=1)
uh = space.function()
bform = BilinearForm(space)
DI = ScalarDiffusionIntegrator(PDE.diffusion_coef(ps))
bform.add_integrator(DI)

lform = LinearForm(space)
SI = ScalarSourceIntegrator(PDE.source(ps))
lform.add_integrator(SI)

A = bform.assembly()
F = lform.assembly()

A, F = DirichletBC(space, gd=PDE.solution).apply(A, F)

uh[:] = cg(A, F, maxit=5000, atol=1e-14, rtol=1e-14)
print(uh[:].shape)
print(PDE.solution(mesh.node).shape)
print(bm.mean(bm.abs(PDE.solution(mesh.node) - uh[:])))

