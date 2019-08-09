import matplotlib.pyplot as plt
import numpy as np
from fenics import *
import dolfin
from guccionematerial import *

parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["cpp_optimize"] = True
parameters['allow_extrapolation'] = True


def load_ellipsoid_data():
    """Returns 4-tuple:
    mesh - the mesh, 
    mf - MeshFunction defining boundary markers, 
    numbering - dict of marking numbers,
    fibers - list of functions defining microstructure"""
  
    mesh = Mesh(MPI.comm_world, "data/mesh.xml")
    mf = MeshFunction("size_t", mesh, "data/facet_function.xml")

    numbering = {
        "BASE": 10,
        "ENDO": 30,
        "EPI": 40
    }

    # load fibers, sheet, cross_sheet data 
    fiber_element = VectorElement(family="Quadrature",
                                     cell=mesh.ufl_cell(),
                                     degree=4,
                                     quad_scheme="default")
    fiber_space = FunctionSpace(mesh, fiber_element)
    fiber = Function(fiber_space, "data/fiber.xml")
    sheet = Function(fiber_space, "data/sheet.xml")
    cross_sheet = Function(fiber_space, "data/cross_sheet.xml")

    fibers = [fiber,sheet, cross_sheet]
    
    return mesh, mf, numbering, fibers

def compute_cavity_volume(mesh,mf, numbering,u=None):
    X = SpatialCoordinate(mesh) 
    N = FacetNormal(mesh)

    if u is not None:
        I = Identity(3) # the identity matrix
        F = I + grad(u) # the deformation gradient
        J = det(F)
        vol_form = (-1.0/3.0) * dot(X + u, J * inv(F).T * N)
    else:
        vol_form = (-1.0/3.0) * dot(X, N)


    ds = Measure('ds',domain=mesh,subdomain_data=mf)

    return assemble(vol_form*ds(numbering["ENDO"]))

mesh, boundary_markers, numbering, fibers = load_ellipsoid_data()

V = VectorFunctionSpace(mesh,'P',1)

# Redefine boundary measure, to allow integration over parts of boundary
ds = Measure('ds',domain=mesh,subdomain_data=boundary_markers)

clamp = Constant((0.0, 0.0, 0.0))
bc = DirichletBC(V, clamp, boundary_markers, numbering["BASE"])
bcs = [bc]

# Define solution u and test function v
u = Function(V)
v = TestFunction(V)

# Define strain measures
I = Identity(3) # the identity matrix
F = I + grad(u) # the deformation gradient
F = variable(F)

mat = GuccioneMaterial(e1=fibers[0],e2=fibers[1],e3=fibers[2],kappa=1e3,Tactive=0.0)

psi = mat.strain_energy(F)
P = diff(psi,F)        # the first Piola-Kirchoff stress tensor

p_endo = Constant(0.0)

# Define nonlinear problem
N = FacetNormal(mesh)
Gext = p_endo * inner(v, det(F)*inv(F)*N) * ds(numbering["ENDO"]) #endocardial pressure
R = inner(P,grad(v))*dx + Gext 

# Step-wise loading 
pressure_steps = 40
target_pressure = 20.0

# Step-wise loading (for plotting and convergence)
active_steps = 40
target_active = 10.0
active = np.linspace(0,target_active,active_steps)

d0 = np.zeros(3)                #displacement at point0
disp = np.zeros(pressure_steps)     #array to store displacement for all steps

u_file = XDMFFile("contraction_august/u.xdmf")
pD1_file = XDMFFile("testing/pD.xdmf")
pnew_file = XDMFFile("testing/pnew.xdmf")
dt = 1/40
t = 0
append = False 

comm = MPI.comm_world
mf = XDMFFile(comm, "data/finer_mesh.xdmf")
finer_mesh = Mesh()
mf.read(finer_mesh)
mf.close()

PP = FunctionSpace(mesh, 'P', 1)
PPP = FunctionSpace(finer_mesh, 'P', 1)
pD1 = Function(PP)
p_new = Function(PPP)

# Loop over load steps:
pressures = np.linspace(0,target_pressure,pressure_steps)
volumes = np.zeros_like(pressures)
for step in range(20):
    p_endo.assign(pressures[step])
    #mat.set_active_stress(active[step])
    solve(R == 0, u, bcs)
    pD1 = project(inner(diff(psi, F), F.T), PP)
    #p_new.set_allow_extrapolation(True)
    p_new = project(pD1, PPP)
    pD1_file.write_checkpoint(pD1, 'pD', step, append = append)
    pnew_file.write_checkpoint(p_new, 'pnew', step, append = append)
    append = True

'''
i = 1

for step in range(20,pressure_steps,1):
    p_endo.assign(pressures[20-i])
    #mat.set_active_stress(active[20-i])
    solve(R == 0, u, bcs)
    pD2 = project(inner(diff(psi, F), F.T), PP)
    #pD.append(p_D1)
    # Compute and store volume for each step:
    #volumes[step] = compute_cavity_volume(mesh,boundary_markers,numbering,u)
    u_file.write_checkpoint(u, 'u', 20+i, append = append)
    pD1_file.write_checkpoint(pD2, 'pD', 20+i, append = append)
    append = True
    i = i+1


p_new = Function(PPP)
p_new = Projection(pD1, PPP)




u_file.close()
pD1_file.close()

plt.figure(1)
plt.scatter(pD)
plt.xlabel('Volume')
plt.ylabel('Pressure')
plt.savefig('contraction_august/p.png')

plt.figure(2)

# Move mesh according to displacement
u_int = interpolate(u, VectorFunctionSpace(mesh, "CG", 1))
moved_mesh = Mesh(mesh)
ALE.move(moved_mesh, u_int)

# Plot the result on to of the original
plot(moved_mesh, alpha=0.1, edgecolor='k', color='w')
plot(mesh, color="r")

ax = plt.gca()
ax.view_init(elev=-67, azim=-179)
ax.set_axis_off()

plt.show()
'''
