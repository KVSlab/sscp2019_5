import matplotlib.pyplot as plt
import numpy as np
from fenics import *
import dolfin
from guccionematerial import *

parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["cpp_optimize"] = True


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
pressure_steps = 20
target_pressure = 10.0

u_file = XDMFFile("u.xdmf")
append = False 

# Loop over load steps:
pressures = np.linspace(0,target_pressure,pressure_steps)
volumes = np.zeros_like(pressures)
for step in range(pressure_steps):
    p_endo.assign(pressures[step])
    solve(R == 0, u, bcs)
    # Compute and store volume for each step:
    volumes[step] = compute_cavity_volume(mesh,boundary_markers,numbering,u)
    u_file.write_checkpoint(u, 'u', step, append = append)
    append = True

u_file.close()

'''
plt.figure(1)
plt.plot(volumes,pressures)
plt.xlabel('Volume')
plt.ylabel('Pressure')
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
