# Importing important libraries
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import ufl
import dolfin

# Parameters for the solver
parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["cpp_optimize"] = True
parameters['allow_extrapolation'] = True


# Loading the mesh: read mesh from xdmf mf1 = finer mesh and mf2 = coarse mesh
comm = MPI.comm_world
mf = XDMFFile(comm, "mesh/finer_mesh.xdmf")
finer_mesh = Mesh()
mf.read(finer_mesh)
mf.close()

mf2 = XDMFFile(comm, "mesh/markers.xdmf")
boundary_markers = MeshFunction('size_t', finer_mesh, finer_mesh.topology().dim()-1)
mf2.read(boundary_markers)
mf2.close()

numbering = {
        "BASE": 10,
        "ENDO": 30,
        "EPI": 40
}

# Loading the coarse mesh and the pressure from it
coarse_mesh = dolfin.Mesh(dolfin.MPI.comm_world, "data/coarse_mesh.xml")
PP = FunctionSpace(coarse_mesh, 'P', 1)
pD = Function(PP)
pD_file = XDMFFile("data/pD.xdmf")  #pressure in the the coarse mesh

# Define mixed function space for Pressure
def mixed_function_space(mesh, N):
    P1 = FiniteElement('P', mesh.ufl_cell(), 1)
    ME = MixedElement([P1 for i in range(N)])
    return FunctionSpace(mesh, ME)

# Define mixed vector function space for Velocity
def mixed_vector_function_space(mesh, N):
    V1 = VectorElement('P', mesh.ufl_cell(), 2)
    ME = MixedElement([V1 for i in range(N)])
    return FunctionSpace(mesh, ME)

#Define mesh and function spaces
N = 5
P = mixed_function_space(finer_mesh, N) # mixed pressure space
#V = VectorFunctionSpace(finer_mesh, 'P', 1)
#W = mixed_vector_function_space(mesh, N)

# Functions and test function for the variational formulation
p = TrialFunction(P)
pX = Function(P.sub(0).collapse())
q = TestFunction(P)

bcs = []

# Define parameters for perfusion for N = 2 to N = 6
if N == 2:
    K = [Constant(1E-3), Constant(20E-3)]   #[m^2/Pa.s]
    beta = [[0., 3E-5], [3E-5, 0.]] #[1/Pa.s]
    f = [Constant(1200), -0.1*(p[-1]-3)]

if N == 3:
    K = [Constant(1E-3), Constant(12E-3), Constant(20E-3)]   #[m^2/Pa.s]
    beta = [[0., 2E-5, 0.], [2E-5, 0., 5E-5], [0., 5E-5, 0.]] #[1/Pa.s]
    f = [Constant(1200), Constant(0.), -0.1*(p[-1]-3)]

if N == 4:
    K = [Constant(1E-3), Constant(12E-3), Constant(16.6E-3), Constant(20E-3)]   #[m^2/Pa.s]
    beta = [[0., 2E-5, 0., 0.], [2E-5, 0., 4E-5, 0.], [0., 4E-5, 0., 4E-5], [0., 0., 4E-5, 0.]] #[1/Pa.s]
    f = [Constant(0.), Constant(0.), Constant(0.), Constant(0.)]

if N == 5:
    K = [Constant(1E-3), Constant(12E-3), Constant(16.6E-3), Constant(18.6E-3), Constant(20E-3)]   #[m^2/Pa.s]
    beta = [[0., 2E-5, 0., 0., 0.], [2E-5, 0., 4E-5, 0., 0.], [0., 4E-5, 0., 4E-5, 0.], [0., 0., 4E-5, 0., 5E-5],[0., 0., 0., 5E-5, 0.]] #[1/Pa.s]
    f = [Constant(0.), Constant(0.), Constant(0.), Constant(0.), Constant(0.)]

if N == 6:
    K = [Constant(1E-3), Constant(12E-3), Constant(16.6E-3), Constant(18.6E-3), Constant(19.4E-3), Constant(20E-3)]   #[m^2/Pa.s]
    beta = [[0., 2E-5, 0., 0., 0., 0.], [2E-5, 0., 4E-5, 0., 0., 0.], [0., 4E-5, 0., 4E-5, 0., 0.], [0., 0., 4E-5, 0., 5E-5, 0.],[0., 0., 0., 5E-5, 0., 5E-5], [0., 0., 0., 0., 5E-5, 0.]] #[1/Pa.s]
    f = [Constant(1200), Constant(0.), Constant(0.), Constant(0.), Constant(0.), -0.1*(p[-1]-3)]

# Define parameters for perfusion
#K = [Constant(1), Constant(12), Constant(16.6), Constant(20)]   #[mm^2/kPa.s]
#beta = [[0., 2E-2, 0., 0.], [2E-2, 0., 4E-2, 0.], [0., 4E-2, 0., 4E-2], [0., 0., 4E-2, 0.]] #[1/kPa.s]

K = [Constant(1E-9), Constant(12E-9), Constant(16.6E-9), Constant(18.6E-9), Constant(20E-9)]   #[m^2/Pa.s]
#beta = [[0., 0.02, 0., 0.], [0.02, 0., 0.05, 0.], [0., 0.05, 0., 0.1], [0., 0.0, 0.1, 0.]]
beta = [[0., 2E-2, 0., 0., 0.], [2E-2, 0., 3E-2, 0., 0.], [0., 3E-2, 0., 4E-2, 0.], [0., 0., 4E-2, 0., 5E-2],[0., 0., 0., 5E-2, 0.]] #[1/kPa.s]
f = [Constant(1200), Constant(0.),Constant(0.),Constant(0.), -0.1*(p[-1]-3)] #[kPa]
phi = [Constant(0.), Constant(0.),Constant(0.), Constant(0.)] #[mm^3/s]

# Define variational problem
R = phi[0]*q[0]*ds(40) + sum([K[i] * inner(grad(p[i]), grad(q[i])) for i in range(N)])*dx(finer_mesh) +\
		beta[0][1]*(pX - p[1])*q[0]*dx - beta[1][0]*(p[1] - pX)*q[1]*dx(finer_mesh) +\
		sum([beta[i+1][i+2]*(p[i+1]-p[i+2])*q[i+1] for i in range(N-2)])*dx(finer_mesh) +\
		sum([beta[i+2][i+1]*(p[i+2]-p[i+1])*q[i+2] for i in range(N-2)])*dx(finer_mesh) -\
		sum([f[i]*q[i] for i in range(N)])*dx(finer_mesh)

a, L = lhs(R), rhs(R)

# Postprocecing velocity
#v = Function(W)
#(v0, v1, v2) = ufl.split(v)

p0_file = XDMFFile("results/p0.xdmf")
p1_file = XDMFFile("results/p1.xdmf")
p2_file = XDMFFile("results/p2.xdmf")
p3_file = XDMFFile("results/p3.xdmf")
p4_file = XDMFFile("results/p4.xdmf")

steps = 40
dt = 1/steps
t = 0
append = False
p = Function(P)
pD = Function(PP)
p_D = Function(P)

for i in range(steps):
	print(i)
	pD_file.read_checkpoint(pD, 'pD', i)
	p_D = project(pD, P.sub(0).collapse())
	pX.assign(p_D)

	# Find solution for pressure in each compartment
	solve(a == L, p, bcs=bcs)
	(p0,p1,p2,p3,p4) = p.split(True)
	
	p0_file.write_checkpoint(p_D, 'p0', t, append = append)
	p1_file.write_checkpoint(p1, 'p1', t, append = append)
	p2_file.write_checkpoint(p2, 'p2', t, append = append)
	p3_file.write_checkpoint(p3, 'p3', t, append = append)
	p4_file.write_checkpoint(p4, 'p4', t, append = append)
	
	append = True
	t += dt
	
p0_file.close()
p1_file.close()
p2_file.close()
p3_file.close()
p4_file.close()
