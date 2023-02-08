from meshpy import triangle
from icepack.meshing import triangle_to_firedrake
from firedrake import CheckpointFile

Lx, Ly = 640e3, 80e3
points = [
    (0, 0),
    (Lx, 0),
    (Lx, Ly),
    (0, Ly)
]

facets = [(i, (i + 1) % len(points)) for i in range(len(points))]
markers = list(range(1, len(points) + 1))

mesh_info = triangle.MeshInfo()
mesh_info.set_points(points)
mesh_info.set_facets(facets, facet_markers=markers)

dy = Ly / 10
area = dy**2 / 2
triangle_mesh = triangle.build(mesh_info, max_volume=area)
coarse_mesh = triangle_to_firedrake(triangle_mesh)

with CheckpointFile("initial_mesh.h5", 'w') as afile:
    afile.save_mesh(coarse_mesh)
    