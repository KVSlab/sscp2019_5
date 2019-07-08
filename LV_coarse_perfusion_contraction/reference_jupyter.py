from fenics import *
import numpy as np
import matplotlib.pyplot as plt

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

from guccionematerial import *

mesh, boundary_markers, numbering, fibers = load_ellipsoid_data()
V = VectorFunctionSpace(mesh, 'P', 1)
W = VectorFunctionSpace(mesh, 'P', 2)

u = Function(V)
u_file = XDMFFile("contraction/u.xdmf")
p0_file = XDMFFile("referencia/p0.xdmf")
p1_file = XDMFFile("referencia/p1.xdmf")
v0_file = XDMFFile("referencia/v0.xdmf")
v1_file = XDMFFile("referencia/v1.xdmf")
append = False

# Define strain measures
u = Function(V)
I = Identity(3) # the identity matrix
F = I + grad(u) # the deformation gradient
F = variable(F)
J = variable(det(F))

mat = GuccioneMaterial(e1=fibers[0],e2=fibers[1],e3=fibers[2],kappa=1e3,Tactive=0.0)
psi = mat.strain_energy(F)

P = FunctionSpace(mesh, 'P', 1) # pressure space

# Define parameters
N = 2
K = [Constant(1), Constant(0.5)]
beta01 = Constant(0.5)
f = Constant(0)

bc_endo = DirichletBC(P, Constant(0), boundary_markers, 30)
bcs = [bc_endo]

# Define variational problem
p = TrialFunction(P)
q = TestFunction(P)
p_D = Function(P)
R = K[1] * inner(grad(p), grad(q))*dx + beta01*(p-p_D)*q*dx - f*q*dx
a, L = lhs(R), rhs(R)

steps = 20
dt = 1/steps
t = 0.0
p = Function(P)
for i in range(steps):
    u_file.read_checkpoint(u, 'u', i)
    
    p_D1 = project(inner(diff(psi, F), F.T), P)
    p_D.assign(p_D1)
    
    solve(a == L, p, bcs=bcs)

    v0 = project(-K[0]*grad(p_D),W)
    v1 = project(-K[0]*grad(p),W)
    
    p0_file.write_checkpoint(p_D, 'p0', t, append=append)
    p1_file.write_checkpoint(p, 'p1', t, append=append)
    v0_file.write_checkpoint(v0, 'p1', t, append=append)
    v1_file.write_checkpoint(v1, 'p1', t, append=append)
    
    append = True
    t += dt
    
u_file.close()
p0_file.close()
p1_file.close()
