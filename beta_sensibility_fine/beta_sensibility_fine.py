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
mf = XDMFFile(comm, "data/finer_mesh.xdmf")
finer_mesh = Mesh()
mf.read(finer_mesh)
mf.close()

# Load markers
mf2 = XDMFFile(comm, "data/markers.xdmf")
boundary_markers = MeshFunction('size_t', finer_mesh, finer_mesh.topology().dim()-1)
mf2.read(boundary_markers)
mf2.close()

# Loading the coarse mesh and the pressure from it
coarse_mesh = dolfin.Mesh(dolfin.MPI.comm_world, "data/coarse_mesh.xml")
PP = FunctionSpace(coarse_mesh, 'P', 1)
pD = Function(PP)
pD_file = XDMFFile("data/pD.xdmf")  #pressure in the the coarse mesh

# Define mixed function space for pressure
def mixed_function_space(mesh, N):
    P1 = FiniteElement('P', mesh.ufl_cell(), 1)
    ME = MixedElement([P1 for i in range(N)])
    return FunctionSpace(mesh, ME)

# Define mixed vector function space for velocity
def mixed_vector_function_space(mesh, N):
    V1 = VectorElement('P', mesh.ufl_cell(), 2)
    ME = MixedElement([V1 for i in range(N)])
    return FunctionSpace(mesh, ME)

#Define mesh and function spaces
N = 3   # min = 2, max = 6
P = mixed_function_space(finer_mesh, N) # mixed pressure space

# Functions and test function for the variational formulation
p = TrialFunctions(P)
pX = Function(P.sub(0).collapse())
q = TestFunctions(P)

bcs = []

# Markers
cells = SubsetIterator(boundary_markers, 40)
number_surface_cells = finer_mesh.num_cells()
source = float(1.2E-6/number_surface_cells)
psi = [Constant(source)] # [m^3/s]

# Define parameters for arteries and capillaries
K_artery = 1E-6                             # default: 1E-6 [m^2/kPa.s]
K_capillary = 20E-6                         # default: 20E-6 [m^2/kPa.s]
beta_artery = 2E-2                          # default: 2E-2 [1/kPa.s]
beta_capillary = 0.5E-2                       # default: 5E-2 [1/kPa.s]
f_source = Constant(.0)                     # default: 1.2E-6 [m^3/s]
f_sink = Constant(0.1)*(p[-1]-Constant(3))  # default: Constant(0.1)*(p[-1]-Constant(3)) [1/s] (pressure in kPa)

# Convert list of float values to list of constant values
def f2c(var):
    for i in range(0,len(var),1):
        var[i] = Constant(var[i])
    return var

# Calculate parameters for perfusion for N = 2 to N = 6
if N == 2:
    K1 = K_artery
    K2 = K_capillary
    beta01 = 3E-2

    K = f2c([K_artery, K_capillary])
    beta = [f2c([0., beta01]), f2c([beta01, 0.])]
    f = [f_source, f_sink]

if N == 3:
    K1 = K_artery
    K3 = K_capillary
    K2 = K1+(K3-K1)/2**(1/3)
    beta01 = beta_artery
    beta12 = beta_capillary

    K = f2c([K1, K2, K3])
    beta = [f2c([0., beta01, 0.]), f2c([beta01, 0., beta12]), f2c([0., beta12, 0.])]
    f = [f_source, Constant(0.), f_sink]

if N == 4:
    K1 = K_artery
    K4 = K_capillary
    K2 = K1+(K4-K1)/2**(1/3)
    K3 = K2+(K4-K2)/2**(1/3)
    beta01 = beta_artery
    beta23 = beta_capillary
    beta12 = beta01+(beta23-beta01)/2**(1/3)

    K = f2c([K1, K2, K3, K4])
    beta = [f2c([0., beta01, 0., 0.]), f2c([beta01, 0., beta12, 0.]), f2c([0., beta12, 0., beta23]), f2c([0., 0., beta23, 0.])]
    f = [f_source, Constant(0.), Constant(0.), f_sink]

if N == 5:
    K1 = K_artery
    K5 = K_capillary
    K2 = K1+(K5-K1)/2**(1/3)
    K3 = K2+(K5-K2)/2**(1/3)
    K4 = K3+(K5-K3)/2**(1/3)
    beta01 = beta_artery
    beta34 = beta_capillary
    beta12 = beta01+(beta34-beta01)/2**(1/3)
    beta23 = beta12+(beta34-beta12)/2**(1/3)

    K = f2c([K1, K2, K3, K4, K5])
    beta = [f2c([0., beta01, 0., 0., 0.]), f2c([beta01, 0., beta12, 0., 0.]), f2c([0., beta12, 0., beta23, 0.]), f2c([0., 0., beta23, 0., beta34]), f2c([0., 0., 0., beta34, 0.])]
    f = [f_source, Constant(0.), Constant(0.), Constant(0.), f_sink]

if N == 6:
    K1 = K_artery
    K6 = K_capillary
    K2 = K1+(K6-K1)/2**(1/3)
    K3 = K2+(K6-K2)/2**(1/3)
    K4 = K3+(K6-K3)/2**(1/3)
    K5 = K4+(K6-K4)/2**(1/3)
    beta01 = beta_artery
    beta45 = beta_capillary
    beta12 = beta01+(beta45-beta01)/2**(1/3)
    beta23 = beta12+(beta45-beta12)/2**(1/3)
    beta34 = beta23+(beta45-beta23)/2**(1/3)

    K = f2c([K1, K2, K3, K4, K5, K6])
    beta = [f2c([0., beta01, 0., 0., 0., 0.]), f2c([beta01, 0., beta12, 0., 0., 0.]), f2c([0., beta12, 0., beta23, 0., 0.]), f2c([0., 0., beta23, 0., beta34, 0.]), f2c([0., 0., 0., beta34, 0., beta45]), f2c([0., 0., 0., 0., beta45, 0.])]
    f = [f_source, Constant(0.), Constant(0.), Constant(0.), Constant(0.), f_sink]

# Define variational problem
if N == 2:
    R = psi[0]*q[0]*ds(40) + sum([K[i] * inner(grad(p[i]), grad(q[i])) for i in range(N)])*dx(finer_mesh) +\
        q[0]*(beta[0][1]*(pX-p[1]))*dx(finer_mesh) +\
        q[1]*(beta[1][0]*(p[1]-pX))*dx(finer_mesh) +\
        sum([f[i]*q[i] for i in range(N)])*dx(finer_mesh)

if N == 3:
    R = psi[0]*q[0]*ds(40) + sum([K[i] * inner(grad(p[i]), grad(q[i])) for i in range(N)])*dx(finer_mesh) +\
        q[0]*(beta[0][1]*(pX-p[1]) + beta[0][2]*(pX-p[2]))*dx(finer_mesh) +\
        q[1]*(beta[1][0]*(p[1]-pX) + beta[1][2]*(p[1]-p[2]))*dx(finer_mesh) +\
        q[2]*(beta[2][0]*(p[2]-pX) + beta[2][1]*(p[2]-p[1]))*dx(finer_mesh) +\
        sum([f[i]*q[i] for i in range(N)])*dx(finer_mesh)

if N == 4:
    R = psi[0]*q[0]*ds(40) + sum([K[i] * inner(grad(p[i]), grad(q[i])) for i in range(N)])*dx(finer_mesh) +\
        q[0]*(beta[0][1]*(pX-p[1]) + beta[0][2]*(pX-p[2]) + beta[0][3]*(pX-p[3]))*dx(finer_mesh) +\
        q[1]*(beta[1][0]*(p[1]-pX) + beta[1][2]*(p[1]-p[2]) + beta[1][3]*(p[1]-p[3]))*dx(finer_mesh) +\
        q[2]*(beta[2][0]*(p[2]-pX) + beta[2][1]*(p[2]-p[1]) + beta[2][3]*(p[2]-p[3]))*dx(finer_mesh) +\
        q[3]*(beta[3][0]*(p[3]-pX) + beta[3][1]*(p[3]-p[1]) + beta[3][2]*(p[3]-p[2]))*dx(finer_mesh) +\
        sum([f[i]*q[i] for i in range(N)])*dx(finer_mesh)

if N == 5:
    R = psi[0]*q[0]*ds(40) + sum([K[i] * inner(grad(p[i]), grad(q[i])) for i in range(N)])*dx(finer_mesh) +\
        q[0]*(beta[0][1]*(pX-p[1]) + beta[0][2]*(pX-p[2]) + beta[0][3]*(pX-p[3]) + beta[0][4]*(pX-p[4]))*dx(finer_mesh) +\
        q[1]*(beta[1][0]*(p[1]-pX) + beta[1][2]*(p[1]-p[2]) + beta[1][3]*(p[1]-p[3]) + beta[1][4]*(p[1]-p[4]))*dx(finer_mesh) +\
        q[2]*(beta[2][0]*(p[2]-pX) + beta[2][1]*(p[2]-p[1]) + beta[2][3]*(p[2]-p[3]) + beta[2][4]*(p[2]-p[4]))*dx(finer_mesh) +\
        q[3]*(beta[3][0]*(p[3]-pX) + beta[3][1]*(p[3]-p[1]) + beta[3][2]*(p[3]-p[2]) + beta[3][4]*(p[3]-p[4]))*dx(finer_mesh) +\
        q[4]*(beta[4][0]*(p[4]-pX) + beta[4][1]*(p[4]-p[1]) + beta[4][2]*(p[4]-p[2]) + beta[4][3]*(p[4]-p[3]))*dx(finer_mesh) +\
        sum([f[i]*q[i] for i in range(N)])*dx(finer_mesh)

if N == 6:
    R = psi[0]*q[0]*ds(40) + sum([K[i] * inner(grad(p[i]), grad(q[i])) for i in range(N)])*dx(finer_mesh) +\
        q[0]*(beta[0][1]*(pX-p[1]) + beta[0][2]*(pX-p[2]) + beta[0][3]*(pX-p[3]) + beta[0][4]*(pX-p[4]) + beta[0][5]*(pX-p[5]))*dx(finer_mesh) +\
        q[1]*(beta[1][0]*(p[1]-pX) + beta[1][2]*(p[1]-p[2]) + beta[1][3]*(p[1]-p[3]) + beta[1][4]*(p[1]-p[4]) + beta[1][5]*(p[1]-p[5]))*dx(finer_mesh) +\
        q[2]*(beta[2][0]*(p[2]-pX) + beta[2][1]*(p[2]-p[1]) + beta[2][3]*(p[2]-p[3]) + beta[2][4]*(p[2]-p[4]) + beta[2][5]*(p[2]-p[5]))*dx(finer_mesh) +\
        q[3]*(beta[3][0]*(p[3]-pX) + beta[3][1]*(p[3]-p[1]) + beta[3][2]*(p[3]-p[2]) + beta[3][4]*(p[3]-p[4]) + beta[3][5]*(p[3]-p[5]))*dx(finer_mesh) +\
        q[4]*(beta[4][0]*(p[4]-pX) + beta[4][1]*(p[4]-p[1]) + beta[4][2]*(p[4]-p[2]) + beta[4][3]*(p[4]-p[3]) + beta[4][5]*(p[4]-p[5]))*dx(finer_mesh) +\
        q[5]*(beta[5][0]*(p[5]-pX) + beta[5][1]*(p[5]-p[1]) + beta[5][2]*(p[5]-p[2]) + beta[5][3]*(p[5]-p[3]) + beta[5][4]*(p[5]-p[4]))*dx(finer_mesh) +\
        sum([f[i]*q[i] for i in range(N)])*dx(finer_mesh)

a, L = lhs(R), rhs(R)

p0_file = XDMFFile("results_N"+str(N)+"/p0.xdmf")
p1_file = XDMFFile("results_N"+str(N)+"/p1.xdmf")
if N > 2:
    p2_file = XDMFFile("results_N"+str(N)+"/p2.xdmf")
if N > 3:
    p3_file = XDMFFile("results_N"+str(N)+"/p3.xdmf")
if N > 4:
    p4_file = XDMFFile("results_N"+str(N)+"/p4.xdmf")
if N > 5:
    p5_file = XDMFFile("results_N"+str(N)+"/p5.xdmf")

steps = 20
dt = 1/steps
t = 0
append = False
p = Function(P)
pD = Function(PP)
p_D = Function(P)

for i in range(steps):
    print(i)
    pD_file.read_checkpoint(pD, 'pD', i)
    p_D = project(pD/100, P.sub(0).collapse())
    pX.assign(p_D)

	# Find solution for pressure in each compartment
    solve(a == L, p, bcs=bcs)

    if N == 2:
        (p0,p1) = p.split(True)
    if N == 3:
        (p0,p1,p2) = p.split(True)
    if N == 4:
        (p0,p1,p2,p3) = p.split(True)
    if N == 5:
        (p0,p1,p2,p3,p4) = p.split(True)
    if N == 6:
        (p0,p1,p2,p3,p4,p5) = p.split(True)
    
    p0_file.write_checkpoint(p_D, 'p0', t, append = append)
    p1_file.write_checkpoint(p1, 'p1', t, append = append)
    if N > 2:
        p2_file.write_checkpoint(p2, 'p2', t, append = append)
    if N > 3: 
        p3_file.write_checkpoint(p3, 'p3', t, append = append)
    if N > 4:
        p4_file.write_checkpoint(p4, 'p4', t, append = append)
    if N > 5:
        p5_file.write_checkpoint(p5, 'p5', t, append = append)

    append = True
    t += dt
	
p0_file.close()
p1_file.close()
if N > 2:
    p2_file.close()
if N > 3:
    p3_file.close()
if N > 4:
    p4_file.close()
if N > 6:
    p5_file.close()