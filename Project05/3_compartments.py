from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from guccionematerial import *

parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["cpp_optimize"] = True

def load_ellipsoid_data():
    """Returns 4-tuple:
    mesh - the mesh, 
    mf - MeshFunction defining boundary markers, 
    numbering - dict of marking numbers,
    fibers - list of functions defining microstructure"""
    import dolfin
  
    mesh = dolfin.Mesh(dolfin.MPI.comm_world, "data/mesh.xml")
    mf = dolfin.MeshFunction("size_t", mesh, "data/facet_function.xml")

    numbering = {
        "BASE": 10,
        "ENDO": 30,
        "EPI": 40
    }

    # load fibers, sheet, cross_sheet data 
    fiber_element = dolfin.VectorElement(family="Quadrature",
                                     cell=mesh.ufl_cell(),
                                     degree=4,
                                     quad_scheme="default")
    fiber_space = dolfin.FunctionSpace(mesh, fiber_element)
    fiber = dolfin.Function(fiber_space, "data/fiber.xml")
    sheet = dolfin.Function(fiber_space, "data/sheet.xml")
    cross_sheet = dolfin.Function(fiber_space, "data/cross_sheet.xml")

    fibers = [fiber,sheet, cross_sheet]
    
    return mesh, mf, numbering, fibers


# Function to create an arbitrary number of mixed space
def mixed_function_space(mesh, N):
    #V1 = VectorElement('P', mesh.ufl_cell(), 1)
    P1 = FiniteElement('P', mesh.ufl_cell(), 1)
    #ME = MixedElement([V1, [P1 for i in range(N)]])
    ME = MixedElement([P1 for i in range(N)])
    return FunctionSpace(mesh, ME)

# load geometry data
mesh, boundary_markers, numbering, fibers = load_ellipsoid_data()

# Define number of compartments
N = 3

V = VectorFunctionSpace(mesh, 'P', 1) #Function Space to store the displacement by the contraction 
P = mixed_function_space(mesh, N-1)   #Mixed function space for perfusion

# Define where to read and write the files
u = Function(V)
u_file = XDMFFile("u.xdmf")
p0_file = XDMFFile("p0.xdmf")
p1_file = XDMFFile("p1.xdmf")
p2_file = XDMFFile("p2.xdmf")
append = False

# Define strain measures
u = Function(V)
I = Identity(3) # the identity matrix
F = I + grad(u) # the deformation gradient
F = variable(F)
J = variable(det(F))

mat = GuccioneMaterial(e1=fibers[0],e2=fibers[1],e3=fibers[2],kappa=1e3,Tactive=0.0)
psi = mat.strain_energy(F)

# Define parameters
K = [Constant(1), Constant(0.5), Constant(0.25)]
beta = [[0., 0.5, 0.], [0.5, 0., 0.25], [0., 0.25, 0.]]
f = [Constant(1), Constant(0.), Constant(0.)]


# Define boundary condition
bc_endo = DirichletBC(P.sub(1), Constant(0), boundary_markers, 30)
bcs = [bc_endo]


# Define variational problem
p = TrialFunction(P)
q = TestFunction(P)
p_D = Function(P)

'''
for i in range(N-1):
	print(i)
	print(p[i+1])
	print(q[i+1])
	print(beta[i+1][i+1])
'''

#Define variational formulation

R0 = beta[0][1]*(p[0]-p_D)*q[0]*dx
#R0 = p_D*q[0]*dx
R1 = sum([K[i] * inner(grad(p[i]), grad(q[i])) for i in range(N-1)])*dx
R2 = sum([beta[i+1][i+2]*(p[i]-p[i+1])*q[i]*dx + beta[i+2][i+1]*(p[i+1]-p[i])*q[i+1]*dx for i in range(N-2)])
a = R0+R1+R2

L = sum([f[i]*q[i] for i in range(N-1)])*dx


# Define function to store the solution of perfusion problem
p = Function(P)

# time step parameters
steps = 20
dt = 1/steps
t = 0.0

# Solution loop
for i in range(steps):
    u_file.read_checkpoint(u, 'u', i)
    
    p_D1 = project(inner(diff(psi, F), F.T), P.sub(0).collapse())  #equation to find the pressure equivalent to the contraction displacement
    p_D.assign(p_D1)
    
    solve(a == L, p, bcs=bcs)
        
    p0_file.write_checkpoint(p_D, 'p0', t, append=append)
    p1_file.write_checkpoint(p1, 'p1', t, append=append)
    p2_file.write_checkpoint(p2, 'p2', t, append=append)
    
    append = True
    t += dt

# Close the files
u_file.close()
p0_file.close()
p1_file.close()
