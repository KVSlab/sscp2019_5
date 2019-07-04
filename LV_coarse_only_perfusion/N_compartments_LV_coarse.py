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

def mixed_function_space(mesh, N):
    P1 = FiniteElement('P', mesh.ufl_cell(), 1)
    ME = MixedElement([P1 for i in range(N)])
    return FunctionSpace(mesh, ME)

# Define parameters
N = 3
K = [Constant(1), Constant(0.5), Constant(0.25)]
beta = [[0., 0.2, 0.], [0.2, 0., 0.3], [0., 0.3, 0.]]
f = [Constant(2.), Constant(2.), Constant(0.)]


#Define mesh and function spaces
mesh, boundary_markers, numbering, fibers = load_ellipsoid_data()
P = mixed_function_space(mesh, N) # mixed pressure space

#Define boundary conditions
bc_epi = DirichletBC(P.sub(0), Constant(10500), boundary_markers, 40)
bc_endo = DirichletBC(P.sub(1), Constant(3300), boundary_markers, 30)
bcs = [bc_epi, bc_endo]

# Define variational problem
p = TrialFunction(P)
q = TestFunction(P)
a = sum([K[i] * inner(grad(p[i]), grad(q[i])) for i in range(N)])*dx +\
		sum([beta[i][i+1]*(p[i]-p[i+1])*q[i] for i in range(N-1)])*dx +\
		sum([beta[i+1][i]*(p[i+1]-p[i])*q[i+1] for i in range(N-1)])*dx

L = sum([f[i]*q[i] for i in range(N)])*dx

#Define storage file
p0_file = XDMFFile("teste_LV3/p0.xdmf")
p1_file = XDMFFile("teste_LV3/p1.xdmf")
p2_file = XDMFFile("teste_LV3/p2.xdmf")

# Solve
p = Function(P)
solve(a == L, p, bcs=bcs)

#Find solutions for each function space
(p0,p1, p2) = p.split(True)

p0_file.write_checkpoint(p0, 'p0')
p1_file.write_checkpoint(p1, 'p1')
p2_file.write_checkpoint(p2, 'p2')


# Plot solution p0
plot(mesh, alpha=0.1, edgecolor='k', color='w')
fig = plot(p0)
plt.colorbar(fig)

ax = plt.gca()
ax.view_init(elev=-67, azim=-179)
ax.set_axis_off()
plt.savefig('teste_LV3/p0.png')

# Plot solution p1
plot(mesh, alpha=0.1, edgecolor='k', color='w')
fig = plot(p1)
plt.colorbar(fig)

ax = plt.gca()
ax.view_init(elev=-67, azim=-179)
ax.set_axis_off()
plt.savefig('teste_LV3/p1.png')

# Plot solution p2
plot(mesh, alpha=0.1, edgecolor='k', color='w')
fig = plot(p2)
plt.colorbar(fig)

ax = plt.gca()
ax.view_init(elev=-67, azim=-179)
ax.set_axis_off()
plt.savefig('teste_LV3/p2.png')
