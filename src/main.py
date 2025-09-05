from mesh import generate_mesh
from material import materials
from fem import assemble_global_stiffness, solve_system
import numpy as np

# Mesh
nodes, elements, material_ids = generate_mesh()

# Assemble
K = assemble_global_stiffness(nodes, elements, materials, material_ids)

# Loads
ndof = len(nodes)*3
f = np.zeros(ndof)
top_nodes = np.where(nodes[:,2] == 1.0)[0]
for n in top_nodes:
    f[3*n+2] = -1000.0

# BCs (fix bottom)
bottom_nodes = np.where(nodes[:,2] == 0.0)[0]
fixed_dofs = []
for n in bottom_nodes:
    fixed_dofs.extend([3*n, 3*n+1, 3*n+2])

# Solve
u = solve_system(K, f, fixed_dofs)
print("Displacements:", u)
