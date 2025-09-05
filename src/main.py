from mesh import generate_mesh
from material import materials
from fem import assemble_global_stiffness, solve_system, compute_stresses
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

# Compute stresses
stresses = compute_stresses(nodes, elements, materials, material_ids, u)
print("\nStress Distribution (at Gauss points for each element):")
for eid, stress in enumerate(stresses):
    print(f"\nElement {eid}:")
    for gp, s in enumerate(stress):
        print(f"  Gauss point {gp}: σ_xx={s[0]:.2e}, σ_yy={s[1]:.2e}, σ_zz={s[2]:.2e}, "
              f"σ_xy={s[3]:.2e}, σ_yz={s[4]:.2e}, σ_zx={s[5]:.2e}")