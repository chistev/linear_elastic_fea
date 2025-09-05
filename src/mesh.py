import numpy as np

def generate_mesh(nx=3, ny=3, nz=3, length=1.0):
    # Coordinates
    x = np.linspace(0, length, nx+1)
    y = np.linspace(0, length, ny+1)
    z = np.linspace(0, length, nz+1)

    nodes = []
    for k in range(nz+1):
        for j in range(ny+1):
            for i in range(nx+1):
                nodes.append([x[i], y[j], z[k]])
    nodes = np.array(nodes)

    # Connectivity
    elements = []
    material_ids = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n0 = i   + j*(nx+1) + k*(nx+1)*(ny+1)
                n1 = n0 + 1
                n3 = n0 + (nx+1)
                n2 = n3 + 1
                n4 = n0 + (nx+1)*(ny+1)
                n5 = n4 + 1
                n7 = n4 + (nx+1)
                n6 = n7 + 1
                elements.append([n0,n1,n2,n3,n4,n5,n6,n7])

                # Assign materials
                material_id = 1 if k == 0 else 2
                material_ids.append(material_id)

    return np.array(nodes), np.array(elements), np.array(material_ids)
