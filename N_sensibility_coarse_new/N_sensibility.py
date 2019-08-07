# Importing important libraries
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import ufl
from guccionematerial import *

# Parameters for the solver
parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["cpp_optimize"] = True

# Loading the mesh: this one is the coarse, but it has fibers
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
N = 4
mesh, boundary_markers, numbering, fibers = load_ellipsoid_data()
P = mixed_function_space(mesh, N) # mixed pressure space
V = VectorFunctionSpace(mesh, 'P', 1)
W = mixed_vector_function_space(mesh, N)

#Define boundary conditions
#bc_epi = DirichletBC(P.sub(0), Constant(10500), boundary_markers, 40)
#bc_endo1 = DirichletBC(P.sub(0), Constant(0), boundary_markers, 30)
#bc_endo2 = DirichletBC(P.sub(1), Constant(0), boundary_markers, 30)
#bc_endo3 = DirichletBC(P.sub(2), Constant(0), boundary_markers, 30)
#bc_endo4 = DirichletBC(P.sub(3), Constant(0), boundary_markers, 30)
#bc_endo5 = DirichletBC(P.sub(4), Constant(0), boundary_markers, 30)
#bc_endo6 = DirichletBC(P.sub(5), Constant(0), boundary_markers, 30)
#bcs3 = [bc_endo1,bc_endo2, bc_endo3]
#bcs4 = [bc_endo1,bc_endo2, bc_endo3, bc_endo4]
#bcs5 = [bc_endo1,bc_endo2, bc_endo3, bc_endo4, bc_endo5]
#bcs6 = [bc_endo1,bc_endo2, bc_endo3, bc_endo4, bc_endo5, bc_endo6]
#bcs = [bc_epi,bc_endo3]
bcs = []

# Functions and test function for the variational formulation
u = Function(V)
p = TrialFunction(P)
pD = Function(P.sub(0).collapse())
q = TestFunction(P)

# Define parameters for perfusion
K = [Constant(2E-9), Constant(2E-9), Constant(2E-9), Constant(2E-9), Constant(2E-9), Constant(2E-9),Constant(2E-9), Constant(2E-9), Constant(2E-9), Constant(2E-9), Constant(2E-9)]   #[m^2/Pa.s]

beta = [[0., 2E-5, 0., 0., 0., 0.], [2E-5, 0., 4E-5, 0., 0., 0.], [0., 4E-5, 0., 6E-5, 0., 0.], [0., 0., 6E-5, 0., 8E-5, 0.],[0., 0., 0., 8E-5, 0., 10E-5], [0., 0., 0., 0., 10E-5, 0.]] #[1/Pa.s]

#beta = [[0., 0.02, 0., 0.], [0.02, 0., 0.05, 0.], [0., 0.05, 0., 0.1], [0., 0.0, 0.1, 0.]]
#beta = [[0., 2E-5, 0., 0., 0., 0.,0.,0.,0.,0.], [3E-5, 0., 3E-5, 0., 0., 0.,0.,0.,0.,0.], [0., 3E-5, 0., 3E-5, 0., 0.,0.,0.,0.,0.], [0., 0., 3E-5, 0., 3E-5, 0.,0.,0.,0.,0.],[0., 0., 0., 3E-5, 0., 3E-5,0.,0.,0.,0.], [0., 0., 0., 0., 3E-5, 0.,3E-5,0.,0.,0.],[0., 0., 0., 0., 0., 3E-5,0.,3E-5,0.,0.],[0., 0., 0., 0., 0., 0.,3E-5,0.,3E-5,0.],[0., 0., 0., 0., 0., 0.,0.,3E-5,0.,3E-5],[0., 0., 0., 0., 0., 0.,0.,0.,3E-5,0.]] #[1/Pa.s]
#f = [Constant(1200), Constant(0.), -0.1*(p[1]-3)]
f = [Constant(0.), Constant(0.), Constant(0.), Constant(0.), Constant(0.), Constant(0.), Constant(0.), Constant(0.), Constant(0.), Constant(0.), Constant(0.)]


# Define variational problem
R = sum([K[i] * inner(grad(p[i]), grad(q[i])) for i in range(N)])*dx(mesh) +\
		beta[0][1]*(pD - p[1])*q[0]*dx - beta[1][0]*(p[1] - pD)*q[1]*dx(mesh) +\
		sum([beta[i+1][i+2]*(p[i+1]-p[i+2])*q[i+1] for i in range(N-2)])*dx(mesh) +\
		sum([beta[i+2][i+1]*(p[i+2]-p[i+1])*q[i+2] for i in range(N-2)])*dx(mesh) -\
		sum([f[i]*q[i] for i in range(N)])*dx(mesh)



a, L = lhs(R), rhs(R)

# Contraction parameters
I = Identity(3) # the identity matrix
F = I + grad(u) # the deformation gradient
F = variable(F)
J = variable(det(F)) # dF
mat = GuccioneMaterial(e1=fibers[0],e2=fibers[1],e3=fibers[2],kappa=1e3,Tactive=0.0)
psi = mat.strain_energy(F) #guccione hyperelastic material energy function

# Postprocecing velocity
v = Function(W)
#(v0, v1, v2,v3) = ufl.split(v)
#(v0, v1, v2, v3) = ufl.split(v)
(v0, v1, v2, v3) = ufl.split(v)

# Reaction (not working yet)
epsilon = 0.1
c = Function(P.sub(1).collapse())
c_n = Function(P.sub(1).collapse())
s = TestFunction(P.sub(1).collapse())

#BCs for reaction
bc_react_epi = DirichletBC(P.sub(1), Constant(1), boundary_markers, 40)
bc_react_endo = DirichletBC(P.sub(1), Constant(0), boundary_markers, 30)
bcs_react = [bc_react_epi, bc_react_endo]

state = 'active'

#Define storage file
u_file = XDMFFile("active/u.xdmf")
p0_file = XDMFFile("testresults_"+str(state)+str(N)+"/p0.xdmf")
p1_file = XDMFFile("testresults_"+str(state)+str(N)+"/p1.xdmf")
p2_file = XDMFFile("testresults_"+str(state)+str(N)+"/p2.xdmf")
p3_file = XDMFFile("testresults_"+str(state)+str(N)+"/p3.xdmf")
p4_file = XDMFFile("testresults_"+str(state)+str(N)+"/p4.xdmf")
p5_file = XDMFFile("testresults_"+str(state)+str(N)+"/p5.xdmf")
p6_file = XDMFFile("results_"+str(state)+str(N)+"/p6.xdmf")
p7_file = XDMFFile("results_"+str(state)+str(N)+"/p7.xdmf")
p8_file = XDMFFile("results_"+str(state)+str(N)+"/p8.xdmf")
p9_file = XDMFFile("results_"+str(state)+str(N)+"/p9.xdmf")
v0_file = XDMFFile("testresults_"+str(state)+str(N)+"/v0.xdmf")
v1_file = XDMFFile("testresults_"+str(state)+str(N)+"/v1.xdmf")
v2_file = XDMFFile("testresults_"+str(state)+str(N)+"/v2.xdmf")
v3_file = XDMFFile("testresults_"+str(state)+str(N)+"/v3.xdmf")
v4_file = XDMFFile("testresults_"+str(state)+str(N)+"/v4.xdmf")
v5_file = XDMFFile("results_"+str(state)+str(N)+"/v5.xdmf")
v6_file = XDMFFile("results_"+str(state)+str(N)+"/v6.xdmf")
v7_file = XDMFFile("results_"+str(state)+str(N)+"/v7.xdmf")
v8_file = XDMFFile("results_"+str(state)+str(N)+"/v8.xdmf")
v9_file = XDMFFile("results_"+str(state)+str(N)+"/v9.xdmf")
#c_file = XDMFFile("results_"+str(state)+str(N)+"/reaction.xdmf") 
append = False

'''
#Sensibility analisys
u_file = XDMFFile("contraction/u.xdmf")
p1_file = XDMFFile("sensibility/p1_"+str(N)+".xdmf")
v1_file = XDMFFile("sensibility/v1_"+str(N)+".xdmf")
append = False
'''
# Solve
steps = 40
dt = 1/steps
t = 0.0
p = Function(P)
n = 0


p1_endo = []
p1_epi = []
p2_endo = []
p2_epi = []
time = []

for i in range(steps):
	# Read the displacement of solid mechanics and project as fluid pressure in compartment 1
	u_file.read_checkpoint(u, 'u', i)
	p_D1 = project(inner(diff(psi, F), F.T), P.sub(0).collapse())
	pD.assign(p_D1)

	# Find solution for pressure in each compartment
	solve(a == L, p, bcs=bcs)
	#Find solutions for each function space
	(p0,p1,p2,p3) = p.split(True)

	p1.set_allow_extrapolation(True)
	p1_epi.append(p1(1,-0.984044,1.20511e-16))
	p1_endo.append(p1(1,-0.686207,8.40362e-17))
	p2.set_allow_extrapolation(True)
	p2_epi.append(p1(1,-0.984044,1.20511e-16))
	p2_endo.append(p1(1,-0.686207,8.40362e-17))
	time.append(t)
	#(po,p1,p2,p3) = p.split(True)

	# Postprocessning for find velocity in each compartment
	v0 = project(-K[0]*grad(pD),W.sub(0).collapse())
	v1 = project(-K[1]*grad(p1),W.sub(1).collapse())
	v2 = project(-K[2]*grad(p2),W.sub(2).collapse())
	v3 = project(-K[3]*grad(p3),W.sub(3).collapse())
	#v4 = project(-K[4]*grad(p4),W.sub(4).collapse())
	#v5 = project(-K[5]*grad(p5),W.sub(5).collapse())
	#v6 = project(-K[6]*grad(p5),W.sub(6).collapse())
	#v7 = project(-K[7]*grad(p5),W.sub(7).collapse())
	#v8 = project(-K[8]*grad(p5),W.sub(8).collapse())
	#v9 = project(-K[9]*grad(p5),W.sub(9).collapse())

	#Z = dot((c - c_n)/dt,s)*dx + dot(dot(v0,grad(c)),s)*dx + dot(epsilon*grad(c),grad(s))*dx - dot(f[0],s)*ds
	#solve (Z==0, c, bcs_react)
	
	# saving solutions
	p0_file.write_checkpoint(pD, 'p0', t, append = append)
	p1_file.write_checkpoint(p1, 'p1', t, append = append)
	p2_file.write_checkpoint(p2, 'p2', t, append = append)
	p3_file.write_checkpoint(p3, 'p3', t, append = append)
	#p4_file.write_checkpoint(p4, 'p4', t, append = append)
	#p5_file.write_checkpoint(p5, 'p5', t, append = append)
	#p6_file.write_checkpoint(p6, 'p6', t, append = append)
	#p7_file.write_checkpoint(p7, 'p7', t, append = append)
	#p8_file.write_checkpoint(p8, 'p8', t, append = append)
	#p9_file.write_checkpoint(p9, 'p9', t, append = append)

	v0_file.write_checkpoint(v0, 'v0', t, append = append)
	v1_file.write_checkpoint(v1, 'v1', t, append = append)
	v2_file.write_checkpoint(v2, 'v2', t, append = append)
	v3_file.write_checkpoint(v3, 'v3', t, append = append)
	#v4_file.write_checkpoint(v4, 'v4', t, append = append)
	#v5_file.write_checkpoint(v5, 'v5', t, append = append)
	#v6_file.write_checkpoint(v6, 'v6', t, append = append)
	#v7_file.write_checkpoint(v7, 'v7', t, append = append)
	#v8_file.write_checkpoint(v8, 'v8', t, append = append)
	#v9_file.write_checkpoint(v9, 'v9', t, append = append)

	#c_file.write_checkpoint(c, 'concentration', t, append = append)

	append = True
	t += dt

fig, (ax,bx) = plt.subplots(2,1)

ax.plot(time, p1_endo, color = 'Red', marker = '*', linestyle = '-', label = 'ENDO')
ax.plot(time, p1_epi, color = 'Blue', marker = 'o', linestyle = '--', label = 'EPI')
ax.set_xlabel('Time')
ax.set_ylabel('Pressure')
ax.legend(loc='best')

bx.plot(time, p1_endo, color = 'Red', marker = '*', linestyle = '-', label = 'ENDO')
bx.plot(time, p1_epi, color = 'Blue', marker = 'o', linestyle = '--', label = 'EPI')
bx.set_xlabel('Time')
bx.set_ylabel('Pressure')
bx.legend(loc='best')

fig.suptitle('Pressure compartment 1')
fig.savefig('pressure1.png')

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
