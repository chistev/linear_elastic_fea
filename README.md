# Finite Element Method (FEM) Analysis System

A Python-based finite element analysis system for 3D linear elastic problems using 8-node hexahedral elements with static and transient dynamic analysis capabilities.

## Overview

This FEM implementation provides a complete workflow for structural analysis including mesh generation, stiffness/mass matrix assembly, transient dynamic analysis using the Newmark-beta method, stress/strain computation, and 3D visualization with PyVista.

## Key Features

### Element Formulation
- **8-Node Hexahedral Elements**: Standard isoparametric formulation for 3D elasticity
- **Gauss Quadrature**: 2×2×2 integration scheme for accurate results
- **B-Matrix Construction**: Full strain-displacement matrix for 3D analysis

### Material Models
- **Isotropic Linear Elasticity**: Hooke's law implementation
- **Multiple Materials**: Support for different material properties per element
- **Material Properties**: Young's modulus, Poisson's ratio, density

### Solvers

**Static Analysis**
- Global stiffness matrix assembly
- Boundary condition application (fixed DOFs)
- Direct solver using sparse linear algebra

**Transient Dynamic Analysis**
- Newmark-beta method (implicit time integration)
- Rayleigh damping (α and β coefficients)
- Time-varying load functions support
- Mass matrix assembly

### Post-Processing

**Stress/Strain Computation**
- Gauss point stress/strain calculation
- Interpolation to nodes for visualization
- All 6 components (σxx, σyy, σzz, σxy, σyz, σzx)

**Visualization with PyVista**
- Deformed mesh visualization
- Displacement vector arrows
- Stress/strain contour plots
- Animation export (MP4)

## Technical Architecture

### Core Modules

**fem.py**
- `element_stiffness_hex8`: Element stiffness matrix (24×24)
- `element_mass_hex8`: Element mass matrix (24×24)
- `assemble_global_stiffness`: Global stiffness matrix assembly
- `assemble_global_mass`: Global mass matrix assembly
- `assemble_damping_matrix`: Rayleigh damping matrix
- `solve_system`: Static solver with boundary conditions
- `solve_transient_system`: Newmark-beta transient solver
- `compute_stresses_and_strains`: Post-processing of strains/stresses

**mesh.py**
- Structured hexahedral mesh generation
- Support for multi-material domains
- Configurable mesh density

**material.py**
- Material class with E, ν, ρ properties
- Predefined materials (Steel, Aluminum)

**main.py**
- Complete simulation workflow
- Visualization pipeline
- Animation creation
