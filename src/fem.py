import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# ----------------------------------------------------------------------
# Element stiffness matrix for 8-node hexahedral element
# ----------------------------------------------------------------------
def element_stiffness_hex8(coords, E, nu):
    """
    Compute stiffness matrix for an 8-node hexahedral element (linear elastic).
    
    coords : (8,3) ndarray
        Nodal coordinates of the element
    E : float
        Young's modulus
    nu : float
        Poisson's ratio

    Returns
    -------
    ke : (24,24) ndarray
        Element stiffness matrix
    """
    # Constitutive matrix (Hooke's law for isotropic 3D elasticity)
    C = E / ((1 + nu)*(1 - 2*nu)) * np.array([
        [1-nu,   nu,    nu,   0,   0,   0],
        [nu,   1-nu,    nu,   0,   0,   0],
        [nu,     nu,  1-nu,   0,   0,   0],
        [0,      0,     0, (1-2*nu)/2, 0, 0],
        [0,      0,     0,   0, (1-2*nu)/2, 0],
        [0,      0,     0,   0,   0, (1-2*nu)/2]
    ])

    # Gauss points (2x2x2)
    gp = [-1/np.sqrt(3), 1/np.sqrt(3)]

    ke = np.zeros((24, 24))

    # Loop over Gauss points
    for xi in gp:
        for eta in gp:
            for zeta in gp:
                # Shape function derivatives wrt natural coords
                dN_dxi = 0.125 * np.array([
                    [-(1-eta)*(1-zeta), -(1-xi)*(1-zeta), -(1-xi)*(1-eta)],
                    [+(1-eta)*(1-zeta), -(1+xi)*(1-zeta), -(1+xi)*(1-eta)],
                    [+(1+eta)*(1-zeta), +(1+xi)*(1-zeta), -(1+xi)*(1+eta)],
                    [-(1+eta)*(1-zeta), +(1-xi)*(1-zeta), -(1-xi)*(1+eta)],
                    [-(1-eta)*(1+zeta), -(1-xi)*(1+zeta), +(1-xi)*(1-eta)],
                    [+(1-eta)*(1+zeta), -(1+xi)*(1+zeta), +(1+xi)*(1-eta)],
                    [+(1+eta)*(1+zeta), +(1+xi)*(1+zeta), +(1+xi)*(1+eta)],
                    [-(1+eta)*(1+zeta), +(1-xi)*(1+zeta), +(1-xi)*(1+eta)]
                ])

                # Jacobian
                J = dN_dxi.T @ coords
                detJ = np.linalg.det(J)
                invJ = np.linalg.inv(J)

                # Derivatives wrt global coordinates
                dN_dx = dN_dxi @ invJ

                # Build B matrix (6x24)
                B = np.zeros((6, 24))
                for i in range(8):
                    Bi = np.array([
                        [dN_dx[i,0], 0, 0],
                        [0, dN_dx[i,1], 0],
                        [0, 0, dN_dx[i,2]],
                        [dN_dx[i,1], dN_dx[i,0], 0],
                        [0, dN_dx[i,2], dN_dx[i,1]],
                        [dN_dx[i,2], 0, dN_dx[i,0]]
                    ])
                    B[:, 3*i:3*i+3] = Bi

                # Integrate stiffness
                ke += (B.T @ C @ B) * detJ

    return ke


# ----------------------------------------------------------------------
# Assembly of global stiffness matrix
# ----------------------------------------------------------------------
def assemble_global_stiffness(nodes, elements, materials, material_ids):
    """
    Assemble global stiffness matrix.
    """
    ndof = len(nodes) * 3
    K = lil_matrix((ndof, ndof))

    for eid, element in enumerate(elements):
        mat = materials[material_ids[eid]]
        ke = element_stiffness_hex8(nodes[element], mat.E, mat.nu)

        # DOF mapping
        dof = []
        for n in element:
            dof.extend([3*n, 3*n+1, 3*n+2])

        # Assembly
        for i in range(24):
            for j in range(24):
                K[dof[i], dof[j]] += ke[i, j]

    return K.tocsr()


# ----------------------------------------------------------------------
# Apply boundary conditions
# ----------------------------------------------------------------------
def apply_boundary_conditions(K, f, fixed_dofs):
    """
    Apply essential boundary conditions by reducing system.
    """
    all_dofs = np.arange(len(f))
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    K_ff = K[free_dofs,:][:,free_dofs]
    f_f = f[free_dofs]

    return K_ff, f_f, free_dofs


# ----------------------------------------------------------------------
# Solve system
# ----------------------------------------------------------------------
def solve_system(K, f, fixed_dofs):
    """
    Solve global system Ku=f with boundary conditions.
    """
    ndof = len(f)
    u = np.zeros(ndof)

    K_ff, f_f, free_dofs = apply_boundary_conditions(K, f, fixed_dofs)
    u_free = spsolve(K_ff, f_f)

    u[free_dofs] = u_free
    return u


# ----------------------------------------------------------------------
# Compute stresses and strains
# ----------------------------------------------------------------------
def compute_stresses_and_strains(nodes, elements, materials, material_ids, u):
    """
    Compute stress and strain distribution at Gauss points for each element.
    
    Parameters
    ----------
    nodes : ndarray
        Nodal coordinates
    elements : ndarray
        Element connectivity
    materials : dict
        Material properties (E, nu) for each material ID
    material_ids : ndarray
        Material ID for each element
    u : ndarray
        Global displacement vector
    
    Returns
    -------
    stresses : list of ndarray
        List of stress tensors (6 components) at Gauss points for each element
    strains : list of ndarray
        List of strain tensors (6 components) at Gauss points for each element
    """
    stresses = []
    strains = []
    gp = [-1/np.sqrt(3), 1/np.sqrt(3)]  # 2x2x2 Gauss points

    for eid, element in enumerate(elements):
        mat = materials[material_ids[eid]]
        E, nu = mat.E, mat.nu
        coords = nodes[element]
        
        # Constitutive matrix
        C = E / ((1 + nu)*(1 - 2*nu)) * np.array([
            [1-nu,   nu,    nu,   0,   0,   0],
            [nu,   1-nu,    nu,   0,   0,   0],
            [nu,     nu,  1-nu,   0,   0,   0],
            [0,      0,     0, (1-2*nu)/2, 0, 0],
            [0,      0,     0,   0, (1-2*nu)/2, 0],
            [0,      0,     0,   0,   0, (1-2*nu)/2]
        ])

        # Element displacements
        dof = []
        for n in element:
            dof.extend([3*n, 3*n+1, 3*n+2])
        u_e = u[dof]  # Element displacement vector (24x1)

        element_stresses = []
        element_strains = []
        # Loop over Gauss points
        for xi in gp:
            for eta in gp:
                for zeta in gp:
                    # Shape function derivatives wrt natural coords
                    dN_dxi = 0.125 * np.array([
                        [-(1-eta)*(1-zeta), -(1-xi)*(1-zeta), -(1-xi)*(1-eta)],
                        [+(1-eta)*(1-zeta), -(1+xi)*(1-zeta), -(1+xi)*(1-eta)],
                        [+(1+eta)*(1-zeta), +(1+xi)*(1-zeta), -(1+xi)*(1+eta)],
                        [-(1+eta)*(1-zeta), +(1-xi)*(1-zeta), -(1-xi)*(1+eta)],
                        [-(1-eta)*(1+zeta), -(1-xi)*(1+zeta), +(1-xi)*(1-eta)],
                        [+(1-eta)*(1+zeta), -(1+xi)*(1+zeta), +(1+xi)*(1-eta)],
                        [+(1+eta)*(1+zeta), +(1+xi)*(1+zeta), +(1+xi)*(1+eta)],
                        [-(1+eta)*(1+zeta), +(1-xi)*(1+zeta), +(1-xi)*(1+eta)]
                    ])

                    # Jacobian
                    J = dN_dxi.T @ coords
                    invJ = np.linalg.inv(J)

                    # Derivatives wrt global coordinates
                    dN_dx = dN_dxi @ invJ

                    # Build B matrix (6x24)
                    B = np.zeros((6, 24))
                    for i in range(8):
                        Bi = np.array([
                            [dN_dx[i,0], 0, 0],
                            [0, dN_dx[i,1], 0],
                            [0, 0, dN_dx[i,2]],
                            [dN_dx[i,1], dN_dx[i,0], 0],
                            [0, dN_dx[i,2], dN_dx[i,1]],
                            [dN_dx[i,2], 0, dN_dx[i,0]]
                        ])
                        B[:, 3*i:3*i+3] = Bi

                    # Compute strain (epsilon = B * u_e)
                    strain = B @ u_e
                    element_strains.append(strain)

                    # Compute stress (sigma = C * epsilon)
                    stress = C @ strain
                    element_stresses.append(stress)

        stresses.append(np.array(element_stresses))  # Shape: (8, 6) for 8 Gauss points
        strains.append(np.array(element_strains))   # Shape: (8, 6) for 8 Gauss points

    return stresses, strains

# ----------------------------------------------------------------------
# Element mass matrix for 8-node hexahedral element
# ----------------------------------------------------------------------
def element_mass_hex8(coords, rho):
    """
    Compute mass matrix for an 8-node hexahedral element.
    
    Parameters
    ----------
    coords : (8,3) ndarray
        Nodal coordinates of the element
    rho : float
        Material density
    
    Returns
    -------
    me : (24,24) ndarray
        Element mass matrix
    """
    # Gauss points (2x2x2)
    gp = [-1/np.sqrt(3), 1/np.sqrt(3)]
    me = np.zeros((24, 24))

    # Shape functions
    for xi in gp:
        for eta in gp:
            for zeta in gp:
                # Shape functions
                N = 0.125 * np.array([
                    (1-xi)*(1-eta)*(1-zeta),
                    (1+xi)*(1-eta)*(1-zeta),
                    (1+xi)*(1+eta)*(1-zeta),
                    (1-xi)*(1+eta)*(1-zeta),
                    (1-xi)*(1-eta)*(1+zeta),
                    (1+xi)*(1-eta)*(1+zeta),
                    (1+xi)*(1+eta)*(1+zeta),
                    (1-xi)*(1+eta)*(1+zeta)
                ])

                # Jacobian
                dN_dxi = 0.125 * np.array([
                    [-(1-eta)*(1-zeta), -(1-xi)*(1-zeta), -(1-xi)*(1-eta)],
                    [+(1-eta)*(1-zeta), -(1+xi)*(1-zeta), -(1+xi)*(1-eta)],
                    [+(1+eta)*(1-zeta), +(1+xi)*(1-zeta), -(1+xi)*(1+eta)],
                    [-(1+eta)*(1-zeta), +(1-xi)*(1-zeta), -(1-xi)*(1+eta)],
                    [-(1-eta)*(1+zeta), -(1-xi)*(1+zeta), +(1-xi)*(1-eta)],
                    [+(1-eta)*(1+zeta), -(1+xi)*(1+zeta), +(1+xi)*(1-eta)],
                    [+(1+eta)*(1+zeta), +(1+xi)*(1+zeta), +(1+xi)*(1+eta)],
                    [-(1+eta)*(1+zeta), +(1-xi)*(1+zeta), +(1-xi)*(1+eta)]
                ])
                J = dN_dxi.T @ coords
                detJ = np.linalg.det(J)

                # Mass matrix contribution: int(rho * N^T * N * detJ)
                N_matrix = np.zeros((3, 24))
                for i in range(8):
                    N_matrix[:, 3*i:3*i+3] = np.eye(3) * N[i]
                me += rho * (N_matrix.T @ N_matrix) * detJ

    return me

# ----------------------------------------------------------------------
# Assemble global mass matrix
# ----------------------------------------------------------------------
def assemble_global_mass(nodes, elements, materials, material_ids):
    """
    Assemble global mass matrix.
    """
    ndof = len(nodes) * 3
    M = lil_matrix((ndof, ndof))

    for eid, element in enumerate(elements):
        mat = materials[material_ids[eid]]
        me = element_mass_hex8(nodes[element], mat.rho)

        # DOF mapping
        dof = []
        for n in element:
            dof.extend([3*n, 3*n+1, 3*n+2])

        # Assembly
        for i in range(24):
            for j in range(24):
                M[dof[i], dof[j]] += me[i, j]

    return M.tocsr()

# ----------------------------------------------------------------------
# Assemble damping matrix (Rayleigh damping)
# ----------------------------------------------------------------------
def assemble_damping_matrix(M, K, alpha, beta):
    """
    Assemble damping matrix using Rayleigh damping: C = alpha*M + beta*K
    """
    return alpha * M + beta * K

# ----------------------------------------------------------------------
# Transient solver using Newmark-beta method
# ----------------------------------------------------------------------
def solve_transient_system(M, C, K, f_t, fixed_dofs, dt, n_steps, gamma=0.5, beta=0.25):
    """
    Solve transient linear elastic system using Newmark-beta method.
    
    Parameters
    ----------
    M : sparse matrix
        Global mass matrix
    C : sparse matrix
        Global damping matrix
    K : sparse matrix
        Global stiffness matrix
    f_t : function
        Time-varying load function, f_t(t) returns load vector
    fixed_dofs : list
        Fixed degrees of freedom
    dt : float
        Time step
    n_steps : int
        Number of time steps
    gamma : float
        Newmark-beta parameter (default 0.5)
    beta : float
        Newmark-beta parameter (default 0.25)
    
    Returns
    -------
    u_t : ndarray
        Displacements at each time step (n_steps, ndof)
    v_t : ndarray
        Velocities at each time step
    a_t : ndarray
        Accelerations at each time step
    """
    ndof = M.shape[0]
    all_dofs = np.arange(ndof)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    # Initialize arrays
    u_t = np.zeros((n_steps, ndof))  # Displacements
    v_t = np.zeros((n_steps, ndof))  # Velocities
    a_t = np.zeros((n_steps, ndof))  # Accelerations

    # Initial conditions (assume zero initial displacement and velocity)
    u = np.zeros(ndof)
    v = np.zeros(ndof)
    a = np.zeros(ndof)

    # Effective stiffness matrix
    K_eff = K[free_dofs, :][:, free_dofs] + (1/(beta*dt**2)) * M[free_dofs, :][:, free_dofs] + (gamma/(beta*dt)) * C[free_dofs, :][:, free_dofs]

    for n in range(n_steps):
        t = n * dt
        f = f_t(t)  # Get load vector at current time

        # Effective force
        f_eff = f[free_dofs] + M[free_dofs, :][:, free_dofs] @ (
            (1/(beta*dt**2)) * u[free_dofs] + (1/(beta*dt)) * v[free_dofs] + (1/(2*beta) - 1) * a[free_dofs]
        ) + C[free_dofs, :][:, free_dofs] @ (
            (gamma/(beta*dt)) * u[free_dofs] + (gamma/beta - 1) * v[free_dofs] + (dt/2) * (gamma/beta - 2) * a[free_dofs]
        )

        # Solve for new displacement
        u_new_free = spsolve(K_eff, f_eff)
        u_new = np.zeros(ndof)
        u_new[free_dofs] = u_new_free

        # Update acceleration and velocity
        a_new = (1/(beta*dt**2)) * (u_new[free_dofs] - u[free_dofs]) - (1/(beta*dt)) * v[free_dofs] - (1/(2*beta) - 1) * a[free_dofs]
        v_new = v[free_dofs] + dt * ((1-gamma) * a[free_dofs] + gamma * a_new)

        # Store results
        u_t[n, :] = u_new
        v_t[n, :] = np.zeros(ndof)
        v_t[n, free_dofs] = v_new
        a_t[n, :] = np.zeros(ndof)
        a_t[n, free_dofs] = a_new

        # Update for next step
        u = u_new
        v = np.zeros(ndof)
        v[free_dofs] = v_new
        a = np.zeros(ndof)
        a[free_dofs] = a_new

    return u_t, v_t, a_t