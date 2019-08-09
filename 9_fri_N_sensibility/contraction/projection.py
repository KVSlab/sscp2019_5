# Importing important libraries
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import ufl


# Loading the mesh: read mesh from xdmf
comm = MPI.comm_world
mf = XDMFFile(comm, "data/mesh.xdmf")
finer_mesh = Mesh()
mf.read(finer_mesh)
mf.close()

#mf2 = XDMFFile(comm, "data/markers.xdmf")
#boundary_markers = MeshFunction('size_t', finer_mesh, mesh.topology().dim()-1)
#mf2.read(boundary_markers)
#mf2.close()

mf3 = XDMFFile(comm, "data/coarse_mesh.xdmf")
coarse_mesh = Mesh()
mf.read(coarse_mesh)
mf.close()


P1 = FunctionSpace(coarse_mesh, 'P', 1)
P2 = FunctionSpace(finer_mesh, 'P', 1)
pD = Function(P1)
p_D = Function(P2)

pD_file = XDMFFile("data/pD.xdmf")
pafter_file = XDMFFile("projection_result/pafter.xdmf")
pbefore_file = XDMFFile("projection_result/pbefore.xdmf")

steps = 40
dt = 1/steps
t = 0
append = False

for i in range(steps):
	print(i)
	pD_file.read_checkpoint(pD, 'pD', i)
	pbefore_file.write_checkpoint(pD, 'pbefore', t, append = append)
	p_D = project(pD, P2)
	pafter_file.write_checkpoint(p_D, 'pafter', t, append = append)
	append = True
	t += dt
	
pafter_file.close()
pbefore_file.close()

