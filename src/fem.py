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