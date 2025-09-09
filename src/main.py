import pyvista as pv
import numpy as np
from mesh import generate_mesh
from material import materials
from fem import assemble_global_stiffness, assemble_global_mass, assemble_damping_matrix, solve_transient_system, compute_stresses_and_strains

# Mesh
nodes, elements, material_ids = generate_mesh()

# Assemble stiffness and mass matrices
K = assemble_global_stiffness(nodes, elements, materials, material_ids)
M = assemble_global_mass(nodes, elements, materials, material_ids)

# Rayleigh damping coefficients
alpha = 0.1
beta = 0.01
C = assemble_damping_matrix(M, K, alpha, beta)

# Time-varying load function
def load_function(t):
    ndof = len(nodes) * 3
    f = np.zeros(ndof)
    top_nodes = np.where(nodes[:, 2] == 1.0)[0]
    for n in top_nodes:
        f[3*n+2] = -1000.0 * np.sin(2 * np.pi * t)
    return f

# Boundary conditions (fix bottom)
bottom_nodes = np.where(nodes[:, 2] == 0.0)[0]
fixed_dofs = []
for n in bottom_nodes:
    fixed_dofs.extend([3*n, 3*n+1, 3*n+2])

# Transient solver parameters
dt = 0.01
n_steps = 100
gamma = 0.5
beta = 0.25

# Solve transient system
u_t, v_t, a_t = solve_transient_system(M, C, K, load_function, fixed_dofs, dt, n_steps)

# Compute stresses and strains at each time step
stresses_t = []
strains_t = []
for n in range(n_steps):
    stresses, strains = compute_stresses_and_strains(nodes, elements, materials, material_ids, u_t[n, :])
    stresses_t.append(stresses)
    strains_t.append(strains)

# --- Visualization with PyVista ---

def interpolate_gauss_to_nodes(nodes, elements, field):
    """
    Interpolate Gauss point field (e.g., stresses/strains) to nodes for visualization.
    field: list of (n_elements, 8, 6) arrays for stresses/strains at 8 Gauss points per element.
    Returns: node_field (n_nodes, 6) with averaged field values at nodes.
    """
    n_nodes = len(nodes)
    node_field = np.zeros((n_nodes, 6))
    node_count = np.zeros(n_nodes)
    
    for eid, element in enumerate(elements):
        # Average the 8 Gauss points' field values for this element
        avg_field = np.mean(field[eid], axis=0)  # Average over 8 Gauss points
        for node_id in element:
            node_field[node_id] += avg_field
            node_count[node_id] += 1
    
    # Normalize by number of elements sharing each node
    node_field[node_count > 0] /= node_count[node_count > 0][:, None]
    return node_field

def create_vtk_mesh(nodes, elements):
    """
    Create a PyVista mesh from nodes and hexahedral elements.
    """
    # VTK cell type for 8-node hexahedron
    cell_type = pv.CellType.HEXAHEDRON
    cells = []
    for element in elements:
        cells.append(8)  # Number of nodes per element
        cells.extend(element)
    cells = np.array(cells)
    grid = pv.UnstructuredGrid(cells, [cell_type] * len(elements), nodes)
    return grid

def visualize_results(nodes, elements, u_t, stresses_t, strains_t, time_step=-1):
    """
    Visualize displacement, stress, and strain fields at a given time step.
    """
    # Create VTK mesh
    grid = create_vtk_mesh(nodes, elements)
    
    # Displacements
    displacements = u_t[time_step].reshape(-1, 3)  # Reshape to (n_nodes, 3)
    grid["displacements"] = displacements
    
    # Interpolate stresses and strains to nodes
    stresses_nodes = interpolate_gauss_to_nodes(nodes, elements, stresses_t[time_step])
    strains_nodes = interpolate_gauss_to_nodes(nodes, elements, strains_t[time_step])
    
    # Add stress and strain components to the mesh
    stress_components = ["sigma_xx", "sigma_yy", "sigma_zz", "sigma_xy", "sigma_yz", "sigma_zx"]
    strain_components = ["epsilon_xx", "epsilon_yy", "epsilon_zz", "epsilon_xy", "epsilon_yz", "epsilon_zx"]
    for i, (s_name, e_name) in enumerate(zip(stress_components, strain_components)):
        grid[s_name] = stresses_nodes[:, i]
        grid[e_name] = strains_nodes[:, i]
    
    # Plotting
    plotter = pv.Plotter()
    
    # Deformed mesh with displacement vectors
    warped = grid.warp_by_vector("displacements", factor=1000)  # Scale for visibility
    plotter.add_mesh(warped, color="white", show_edges=True, opacity=0.5)
    plotter.add_arrows(grid.points, displacements, mag=1000, color="red")
    
    # Scalar bar for a stress component (e.g., sigma_xx)
    plotter.add_mesh(grid, scalars="sigma_xx", cmap="viridis", show_edges=True)
    plotter.add_scalar_bar(title="σ_xx (Pa)")
    
    plotter.show()

def animate_transient_results(nodes, elements, u_t, stresses_t, strains_t, filename="transient_simulation.mp4"):
    """
    Create an animation of the transient results.
    """
    grid = create_vtk_mesh(nodes, elements)
    plotter = pv.Plotter(off_screen=True)
    
    # Open movie file
    plotter.open_movie(filename)
    
    for t in range(len(u_t)):
        # Update displacements
        displacements = u_t[t].reshape(-1, 3)
        grid["displacements"] = displacements
        
        # Update stresses
        stresses_nodes = interpolate_gauss_to_nodes(nodes, elements, stresses_t[t])
        grid["sigma_xx"] = stresses_nodes[:, 0]
        
        # Clear plotter and add deformed mesh
        plotter.clear()
        warped = grid.warp_by_vector("displacements", factor=1000)
        plotter.add_mesh(warped, scalars="sigma_xx", cmap="viridis", show_edges=True)
        plotter.add_scalar_bar(title="σ_xx (Pa)")
        
        # Write frame
        plotter.write_frame()
    
    plotter.close()

# Visualize final time step
visualize_results(nodes, elements, u_t, stresses_t, strains_t, time_step=-1)

# Create animation for transient results
animate_transient_results(nodes, elements, u_t, stresses_t, strains_t, filename="transient_simulation.mp4")
