# GEP Dataset Generation

This project provides a suite of scripts to generate large-scale datasets for various physical systems governed by Partial Differential Equations (PDEs). It focuses on Generalized Eigenvalue Problems (GEP) and uses MPI with PETSc/SLEPc for high-performance, parallel computation. The covered simulations include:

* EGFR Electronic Structure
* Electromagnetic Cavity Resonance
* Kirchhoff-Love Plate Vibration
* 2D Thermal Diffusion
* Piezoelectric Coupled-Fields

---

## 1. Prerequisites

Before you begin, ensure you have the following installed:

* A working MPI implementation (e.g., Open MPI, MPICH)
* PETSc
* SLEPc
* Python 3 with the `mpi4py` and `scipy` packages

---

## 2. Environment Setup (`setup_env.sh`)

Before running any generation scripts, you must configure the runtime environment for PETSc and SLEPc. The `setup_env.sh` script handles the setup of all necessary environment variables.

### 2.1 How to Use

In your terminal, you **must** execute this script using the `source` command to ensure that the environment variables are applied to your current session.

```bash
source setup_env.sh
```

---

## 3. Dataset Generation

All datasets are generated in parallel using `mpirun`.  
The general command format is:

```bash
mpirun -np <num_processes> python3 datasets/<script_name>.py --out <output_dir> --num <num_samples> --nx <X_dim> --ny <Y_dim> --nev <num_eigenvalues>
```

- `<num_processes>`: Recommended to set equal to the number of CPU cores on your machine (e.g., 4 or 8).  
- `<script_name>`: The Python script corresponding to the dataset you want to generate.  
- `<output_dir>`: The root directory where the generated dataset will be saved.  
- `<num_samples>`: Total number of samples to generate.  
- `<X_dim>` / `<Y_dim>`: Defines the discretized mesh size of the problem domain. The final matrix size is `nx * ny`.  
- `<num_eigenvalues>`: The number of smallest eigenvalues to solve for each sample.  

---

### 3.1 EGFR Dataset

Generate 100 samples, each with a matrix size of 50000 × 50000, solving for the 500 smallest eigenvalues:

```bash
mpirun -np 4 python3 datasets/build_egfr_parallel.py --out data/egfr_dataset --num 100 --nbas 50000 --nev 500
```

### 3.2 Electromagnetic Cavity (EM Cavity) Dataset

```bash
mpirun -np 4 python3 datasets/build_em_cavity.py --out data/em_cavity_dataset --num 100 --nx 100 --ny 100 --nev 500
```

### 3.3 Plate Vibration Dataset

```bash
mpirun -np 4 python3 datasets/build_plate.py --out data/plate_dataset --num 100 --nx 100 --ny 100 --nev 500
```

### 3.4 Thermal Diffusion Dataset

```bash
mpirun -np 4 python3 datasets/build_thermal.py --out data/thermal_dataset --num 100 --nx 100 --ny 100 --nev 500
```

### 3.5 Piezoelectric Coupled Field Dataset

```bash
mpirun -np 4 python3 datasets/build_piezo.py --out data/piezo_dataset --num 100 --nx 71 --ny 71 --nev 500 --no_solve
```

---

## 4. Output Structure

After successful execution, each dataset will be organized in the specified output directory with the following structure:

```bash
<output_dir>/
├── meta.json              # Metadata of the dataset
├── <global_matrix>.npz    # (Optional) Shared matrix across all samples
└── 00000/                 # First sample
    ├── A.npz              # Sample-specific matrix A
    ├── B.npz              # Sample-specific matrix B
    ├── eigvals.npy        # Computed eigenvalues (if solved)
    └── eigvecs.npy        # Computed eigenvectors (if solved)
```

---

## 5. Mathematical Models and Problem Settings

This section provides the detailed mathematical formulation, input parameters, physical meanings, and algorithmic implementation notes for each dataset problem.

### 5.1 Kirchhoff–Love Plate Vibration

**Governing PDE**  
The free vibration of thin plates is governed by the biharmonic equation:

$$
D \Delta^2 w(x,y) = \rho h \, \frac{\partial^2 w(x,y,t)}{\partial t^2},
$$

with displacement 

$$
w(x,y,t) = \phi(x,y)\cos(\omega t).
$$

This leads to the eigenvalue problem:

$$
D \Delta^2 \phi(x,y) = \lambda \phi(x,y), \quad \lambda = \rho h \omega^2.
$$

**Matrix Formulation (after FEM discretization)**  

$$
A u = \lambda M u
$$

-   $A$: stiffness matrix (assembled from biharmonic operator)
-   $M$: mass matrix
-   $u$: discretized displacement vector
-   $\lambda$: squared natural frequency ($\rho h \omega^2$)

**Input Parameters & Physical Meaning**  
-   $E(x,y)$: Young’s modulus field (elastic property)
-   $\nu$: Poisson’s ratio (dimensionless coupling constant)
-   $h(x,y)$: plate thickness distribution
-   $\rho(x,y)$: density field

Boundary conditions: clamped (Dirichlet: displacement fixed at boundary).

**Implementation Note**  
- Mixed FEM with auxiliary variable $\psi = \Delta w$.
- Assembled into generalized eigenvalue problem $Au = \lambda M u$.

---

### 5.2 EGFR Electronic Structure (Quantum Hamiltonian)

**Governing PDE (Hartree-Fock-Roothaan Equation)**
The electronic structure is described by the time-independent Schrödinger equation, which takes the following form in the Hartree-Fock approximation:

$$
\left(-\frac{1}{2}\nabla^{2}+V_{eff}(r)\right)\psi_{i}(r)=\epsilon_{i}\psi_{i}(r), \quad r\in\mathbb{R}^{3}
$$

-   $\psi_{i}(r)$: The spatial orbital function for the $i$-th electron (the eigenfunction).
-   $\epsilon_{i}$: The energy corresponding to the $i$-th orbital (the eigenvalue).
-   $V_{eff}(r)$: The effective potential energy field, which includes nuclear-electron Coulomb potential, electron-electron repulsion, and exchange-correlation potential.
-   $\nabla^{2}$: The Laplacian operator representing the kinetic energy term. The $-\frac{1}{2}$ factor is a result of using atomic units.

**Matrix Formulation (after discretization)**
After spatial discretization using a method like the Finite Difference Method (FDM), the PDE is converted into a generalized eigenvalue problem:

$$
A\psi_{i}=\epsilon_{i}M\psi_{i}
$$

-   $A \in \mathbb{R}^{n \times n}$: The discretized Hamiltonian matrix, often represented as a sum of the kinetic energy matrix ($T$) and the diagonal potential energy matrix ($V$).
-   $M \in \mathbb{R}^{n \times n}$: The mass matrix. For FDM, this is typically the identity matrix.
-   $\psi_{i} \in \mathbb{R}^{n}$: The discrete eigenvector for the $i$-th orbital.
-   $\epsilon_{i}$: The eigenvalue corresponding to the $i$-th orbital's energy.

**Input Parameters & Physical Meaning**
-   $Z_{i}$: The atomic number for the $i$-th atom (e.g., Hydrogen=1, Carbon=6), which defines the nuclear charge.
-   $R_{i} \in \mathbb{R}^{3}$: The 3D coordinates of each atom, defining the molecular geometry.
-   $N$: The total number of atoms, which determines the overall size and complexity of the system.
-   Total charge / number of electrons: This determines the electron filling arrangement for the self-consistent field solution.

**Implementation Note**
The implementation involves discretizing the Schrödinger equation over a spatial grid using the Finite Difference Method (FDM). This transforms the differential equation into a matrix eigenvalue problem $A\psi = \epsilon M \psi$, which is then solved to find the orbital energies and wavefunctions.

---

### 5.3 Electromagnetic Cavity Resonance (TE Modes)

**Governing PDE (2D TE case)**  

$$
\nabla \cdot \left( \frac{1}{\mu(x,y)} \nabla E_z(x,y) \right) = -\lambda \, \epsilon(x,y) E_z(x,y),
\quad \lambda = \omega^2,
$$

where $E_z(x,y)$ is the scalar out-of-plane electric field.

**Matrix Formulation**  

$$
A e = \lambda M e
$$

-   Stiffness Matrix:
    ```math
    A_{ij} = \int_\Omega \frac{1}{\mu} \nabla \phi_i \cdot \nabla \phi_j \, dx
-   Mass Matrix:
    ```math
    M_{ij} = \int_\Omega \epsilon \, \phi_i \phi_j \, dx

**Input Parameters & Physical Meaning**  
-   $\epsilon(x,y)$: permittivity distribution (affects resonance modes)
-   $\mu(x,y)$: permeability distribution (often constant $\mu_0$)

**Implementation Note**  
- Linear FEM discretization on rectangular mesh.  
- Eigenvalue problem solved with Krylov-Schur / JD / LOBPCG solvers.  
 

---

### 5.4 Piezoelectric Coupled-Field Modes

**Governing Equations**  
Piezoelectric materials couple mechanical strain and electric displacement. The linearized constitutive relations are:

$$
\sigma_{ij} = c_{ijkl} \varepsilon_{kl} - e_{kij} E_k,
$$

$$
D_i = e_{ikl} \varepsilon_{kl} + \epsilon_{ij} E_j,
$$

where stress $\sigma$, strain $\varepsilon$, electric field $E$, and displacement $D$ interact.

**Generalized Eigenvalue Problem (after FEM discretization):**

$$
\begin{bmatrix}
K_{uu} & K_{u\phi} \\
K_{\phi u} & K_{\phi\phi}
\end{bmatrix}
\begin{bmatrix}
u \\
\phi
\end{bmatrix}
= \omega^{2}
\begin{bmatrix}
M_{uu} & 0 \\
0 & 0
\end{bmatrix}
\begin{bmatrix}
u \\
\phi
\end{bmatrix}
$$

-   $K_{uu}$: structural stiffness matrix
-   $K_{u\phi}, K_{\phi u}$: piezoelectric coupling matrices
-   $K_{\phi\phi}$: dielectric stiffness matrix
-   $M_{uu}$: mass matrix
-   $u, \phi$: vectors of mechanical and electrical degrees of freedom

**Input Parameters & Physical Meaning**  
-   $c_{ijkl}(x,y)$: elastic stiffness tensor
-   $e_{ijk}(x,y)$: piezoelectric coupling tensor
-   $\epsilon_{ij}(x,y)$: dielectric tensor
-   $\rho(x,y)$: density distribution
-   Geometry & boundary conditions: free/constrained edges, electrodes

**Implementation Note**  
- Mixed FEM formulation with mechanical and electrical DOFs.  
- Assembled into large block matrix system.  

---

### 5.5 Thermal Diffusion Eigenmodes

**Governing PDE**  

$$
- \nabla \cdot \big( k(x,y) \nabla u(x,y) \big) = \lambda c(x,y) u(x,y),
$$

where $u(x,y)$ is the temperature mode shape, $\lambda$ is the decay rate.

**Matrix Formulation**  

$$
K u_h = \lambda M u_h
$$

-   $K$: stiffness matrix from thermal conductivity
-   $M$: mass matrix from heat capacity density

**Input Parameters & Physical Meaning**  
-   $k(x,y)$: thermal conductivity (heat transport ability)
-   $c(x,y)$: volumetric heat capacity (thermal inertia)

**Implementation Note**  
- FEM discretization of heat operator.  
- Eigenmodes correspond to spatial thermal decay patterns.
---
