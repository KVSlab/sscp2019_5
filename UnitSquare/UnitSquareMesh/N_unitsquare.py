from fenics import *
import numpy as np
import matplotlib.pyplot as plt

parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["cpp_optimize"] = True

def mixed_function_space(mesh, N):
    P1 = FiniteElement('P', mesh.ufl_cell(), 1)
    ME = MixedElement([P1 for i in range(N)])
    return FunctionSpace(mesh, ME)


# Create mesh and define function space
n = 16
mesh = UnitSquareMesh(n, n)
N = 5
P = mixed_function_space(mesh, N)

# Define parameters
i = 1	
K = [Constant(2), Constant(1), Constant(3), Constant(4), Constant(10)]
beta = [[0., 0.2, 0., 0., 0.], [0.5, 0., 0.25, 0., 0.], [0., 0.95, 0., 0.3, 0.], [0., 0., 0.3, 0., 0.1], [0., 0., 0., 0.3, 0.]]
f = [Constant(2.), Constant(2.), Constant(2.), Constant(2.), Constant(0.)]	

# Define boundary condition
def left(x, on_boundary):
    return on_boundary and near(x[0], 0.0)

def right(x, on_boundary):
    return on_boundary and near(x[0], 1.0)

bcl = DirichletBC(P.sub(0), Constant(1.0), left)
bcr1 = DirichletBC(P.sub(1), Constant(0.0), right)
bcr2 = DirichletBC(P.sub(2), Constant(0.0), right)
bcr3 = DirichletBC(P.sub(3), Constant(0.0), right)
bcr4 = DirichletBC(P.sub(4), Constant(0.0), right)
bcs = [bcl, bcr1, bcr2, bcr3, bcr4]

# Define variational problem
p = TrialFunction(P)
q = TestFunction(P)

a = sum([K[i] * inner(grad(p[i]), grad(q[i])) for i in range(N)])*dx +\
		sum([beta[i][i+1]*(p[i]-p[i+1])*q[i] for i in range(N-1)])*dx +\
		sum([beta[i+1][i]*(p[i+1]-p[i])*q[i+1] for i in range(N-1)])*dx

L = sum([f[i]*q[i] for i in range(N)])*dx

# Solve
p = Function(P)
solve(a == L, p, bcs=bcs)

(p1, p2, p3, p4, p5) = p.split(True)

p1File = File('teste3/p1_1.pvd') << p1
p2File = File('teste3/p2_1.pvd') << p2
p3File = File('teste3/p3_1.pvd') << p3
p4File = File('teste3/p4_1.pvd') << p4
p5File = File('teste3/p5_1.pvd') << p5
