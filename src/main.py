from mesh import generate_mesh
from material import materials
from fem import assemble_global_stiffness, assemble_global_mass, assemble_damping_matrix, solve_transient_system, compute_stresses_and_strains
import numpy as np

# Mesh
nodes, elements, material_ids = generate_mesh()

# Assemble stiffness and mass matrices
K = assemble_global_stiffness(nodes, elements, materials, material_ids)
M = assemble_global_mass(nodes, elements, materials, material_ids)

# Rayleigh damping coefficients
alpha = 0.1  # Mass-proportional damping
beta = 0.01  # Stiffness-proportional damping
C = assemble_damping_matrix(M, K, alpha, beta)

# Time-varying load function
def load_function(t):
    ndof = len(nodes) * 3
    f = np.zeros(ndof)
    top_nodes = np.where(nodes[:, 2] == 1.0)[0]
    for n in top_nodes:
        f[3*n+2] = -1000.0 * np.sin(2 * np.pi * t)  # Sinusoidal load
    return f

# Boundary conditions (fix bottom)
bottom_nodes = np.where(nodes[:, 2] == 0.0)[0]
fixed_dofs = []
for n in bottom_nodes:
    fixed_dofs.extend([3*n, 3*n+1, 3*n+2])

# Transient solver parameters
dt = 0.01  # Time step (s)
n_steps = 100  # Number of time steps
gamma = 0.5  # Newmark-beta parameter
beta = 0.25  # Newmark-beta parameter

# Solve transient system
u_t, v_t, a_t = solve_transient_system(M, C, K, load_function, fixed_dofs, dt, n_steps)

# Compute stresses and strains at each time step
stresses_t = []
strains_t = []
for n in range(n_steps):
    stresses, strains = compute_stresses_and_strains(nodes, elements, materials, material_ids, u_t[n, :])
    stresses_t.append(stresses)
    strains_t.append(strains)

# Output results (example: at final time step)
print("\nDisplacements at final time step:")
print(u_t[-1, :])

print("\nStress Distribution at final time step (Gauss points for each element):")
for eid, stress in enumerate(stresses_t[-1]):
    print(f"\nElement {eid}:")
    for gp, s in enumerate(stress):
        print(f"  Gauss point {gp}: σ_xx={s[0]:.2e}, σ_yy={s[1]:.2e}, σ_zz={s[2]:.2e}, "
              f"σ_xy={s[3]:.2e}, σ_yz={s[4]:.2e}, σ_zx={s[5]:.2e}")

print("\nStrain Distribution at final time step (Gauss points for each element):")
for eid, strain in enumerate(strains_t[-1]):
    print(f"\nElement {eid}:")
    for gp, e in enumerate(strain):
        print(f"  Gauss point {gp}: ε_xx={e[0]:.2e}, ε_yy={e[1]:.2e}, ε_zz={e[2]:.2e}, "
              f"ε_xy={e[3]:.2e}, ε_yz={e[4]:.2e}, ε_zx={e[5]:.2e}")