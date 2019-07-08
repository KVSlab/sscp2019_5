from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import ufl

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

def mixed_vector_function_space(mesh, N):
    V1 = VectorElement('P', mesh.ufl_cell(), 2)
    ME = MixedElement([V1 for i in range(N)])
    return FunctionSpace(mesh, ME)

#Define mesh and function spaces
N = 3
mesh, boundary_markers, numbering, fibers = load_ellipsoid_data()
P = mixed_function_space(mesh, N) # mixed pressure space
V = VectorFunctionSpace(mesh, 'P', 1)
W = mixed_vector_function_space(mesh, N)

#Define boundary conditions
bc_epi = DirichletBC(P.sub(0), Constant(10500), boundary_markers, 40)
bc_endo1 = DirichletBC(P.sub(0), Constant(0), boundary_markers, 30)
bc_endo2 = DirichletBC(P.sub(1), Constant(0), boundary_markers, 30)
bc_endo3 = DirichletBC(P.sub(2), Constant(0), boundary_markers, 30)
bcs = [bc_endo1,bc_endo2,bc_endo3]
#bcs = [bc_epi,bc_endo3]

# Functions
u = Function(V)
p = TrialFunction(P)
pD = Function(P.sub(0).collapse())
q = TestFunction(P)

# Define parameters
K = [Constant(0.5), Constant(1.), Constant(2.5)]
beta = [[0., 0.5, 0.], [0.5, 0., 0.5], [0., 0.5, 0.]]
#f = [Constant(1200), Constant(0.), -0.1*(p[1]-3)]
f = [Constant(0.), Constant(0.), Constant(0.)]


# Define variational problem
R = sum([K[i] * inner(grad(p[i]), grad(q[i])) for i in range(N)])*dx +\
		beta[0][1]*(pD - p[1])*q[0]*dx - beta[1][0]*(p[1] - pD)*q[1]*dx +\
		sum([beta[i+1][i+2]*(p[i+1]-p[i+2])*q[i+1] for i in range(N-2)])*dx +\
		sum([beta[i+2][i+1]*(p[i+2]-p[i+1])*q[i+2] for i in range(N-2)])*dx -\
		sum([f[i]*q[i] for i in range(N)])*dx

a, L = lhs(R), rhs(R)

# Contraction
from guccionematerial import *
I = Identity(3) # the identity matrix
F = I + grad(u) # the deformation gradient
F = variable(F)
J = variable(det(F))
mat = GuccioneMaterial(e1=fibers[0],e2=fibers[1],e3=fibers[2],kappa=1e3,Tactive=0.0)
psi = mat.strain_energy(F)

# Postprocecing velocity
v = Function(W)
(v0, v1, v2) = ufl.split(v)

# Reaction
epsilon = 0.1
c = Function(P.sub(1).collapse())
c_n = Function(P.sub(1).collapse())
s = TestFunction(P.sub(1).collapse())

bc_react_epi = DirichletBC(P.sub(1), Constant(1), boundary_markers, 40)
bc_react_endo = DirichletBC(P.sub(1), Constant(0), boundary_markers, 30)
bcs_react = [bc_react_epi, bc_react_endo]

#Define storage file
u_file = XDMFFile("contraction/u.xdmf")
p0_file = XDMFFile("results/p0.xdmf")
p1_file = XDMFFile("results/p1.xdmf")
p2_file = XDMFFile("results/p2.xdmf")
v0_file = XDMFFile("results/v0.xdmf")
v2_file = XDMFFile("results/v1.xdmf")
v1_file = XDMFFile("results/v2.xdmf")
c_file = XDMFFile("results/reaction.xdmf") 
append = False

# Solve
steps = 20
dt = 1/steps
t = 0.0
p = Function(P)
for i in range(steps):
	u_file.read_checkpoint(u, 'u', i)
	p_D1 = project(inner(diff(psi, F), F.T), P.sub(0).collapse())
	pD.assign(p_D1)

	solve(a == L, p, bcs=bcs)
	#Find solutions for each function space
	(p0,p1, p2) = p.split(True)

	v0 = project(-K[1]*grad(pD),W.sub(0).collapse())
	v1 = project(-K[1]*grad(p1),W.sub(1).collapse())
	v2 = project(-K[2]*grad(p2),W.sub(2).collapse())

	#Z = dot((c - c_n)/dt,s)*dx + dot(dot(v0,grad(c)),s)*dx + dot(epsilon*grad(c),grad(s))*dx - dot(f[0],s)*ds
	#solve (Z==0, c, bcs_react)
	
	p0_file.write_checkpoint(pD, 'p0', t, append = append)
	p1_file.write_checkpoint(p1, 'p1', t, append = append)
	p2_file.write_checkpoint(p2, 'p2', t, append = append)

	v0_file.write_checkpoint(v0, 'v0', t, append = append)
	v1_file.write_checkpoint(v1, 'v1', t, append = append)
	v2_file.write_checkpoint(v2, 'v2', t, append = append)

	c_file.write_checkpoint(c, 'concentration', t, append = append)

	append = True
	t += dt

'''
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
'''
