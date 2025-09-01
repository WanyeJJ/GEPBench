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

## 5. Mathematical Models and Problem Settings

This section provides the detailed mathematical formulation, input parameters, physical meanings, and algorithmic implementation notes for each dataset problem.

### 5.1 Kirchhoff–Love Plate Vibration

**Governing PDE**  
[cite_start]The free vibration of thin plates is governed by the biharmonic equation[cite: 3]:

$$
D \Delta^2 w(x,y) = \rho h \, \frac{\partial^2 w(x,y,t)}{\partial t^2},
$$

with displacement 

$$
w(x,y,t) = \phi(x,y)\cos(\omega t).
$$
 [cite_start][cite: 15]
[cite_start]This leads to the eigenvalue problem[cite: 15]:

$$
D \Delta^2 \phi(x,y) = \lambda \phi(x,y), \quad \lambda = \rho h \omega^2.
$$
 [cite_start][cite: 16]
**Matrix Formulation (after FEM discretization)**  

$$
A u = \lambda M u
$$
 [cite_start][cite: 22]
- [cite_start]\(A\): stiffness matrix (assembled from biharmonic operator) [cite: 24]  
- [cite_start]\(M\): mass matrix [cite: 25]  
- [cite_start]\(u\): discretized displacement vector [cite: 26]  
- [cite_start]\(\lambda\): squared natural frequency (\(\rho h \omega^2\)) [cite: 27]  

**Input Parameters & Physical Meaning**  
- [cite_start]\(E(x,y)\): Young’s modulus field (elastic property) [cite: 29]  
- [cite_start]\(\nu\): Poisson’s ratio (dimensionless coupling constant) [cite: 29]  
- [cite_start]\(h(x,y)\): plate thickness distribution [cite: 29]  
- [cite_start]\(\rho(x,y)\): density field [cite: 29]  

[cite_start]Boundary conditions: clamped (Dirichlet: displacement fixed at boundary)[cite: 30].

**Implementation Note**  
- [cite_start]Mixed FEM with auxiliary variable \(\psi = \Delta w\). [cite: 17]  
- [cite_start]Assembled into generalized eigenvalue problem \(Au = \lambda M u\). [cite: 21, 22]  

---

### 5.2 EGFR Electronic Structure (Quantum Hamiltonian)

**Governing PDE (Hartree-Fock-Roothaan Equation)**
[cite_start]The electronic structure is described by the time-independent Schrödinger equation [cite: 34][cite_start], which takes the following form in the Hartree-Fock approximation[cite: 34]:

$$
\left(-\frac{1}{2}\nabla^{2}+V_{eff}(r)\right)\psi_{i}(r)=\epsilon_{i}\psi_{i}(r), \quad r\in\mathbb{R}^{3}
$$
 [cite_start][cite: 35]
-   [cite_start]$\psi_{i}(r)$: The spatial orbital function for the $i$-th electron (the eigenfunction)[cite: 36].
-   [cite_start]$\epsilon_{i}$: The energy corresponding to the $i$-th orbital (the eigenvalue)[cite: 37].
-   [cite_start]$V_{eff}(r)$: The effective potential energy field, which includes nuclear-electron Coulomb potential, electron-electron repulsion, and exchange-correlation potential[cite: 38, 39, 40, 41].
-   [cite_start]$\nabla^{2}$: The Laplacian operator representing the kinetic energy term[cite: 42]. [cite_start]The $-\frac{1}{2}$ factor is a result of using atomic units[cite: 43].

**Matrix Formulation (after discretization)**
[cite_start]After spatial discretization using a method like the Finite Difference Method (FDM), the PDE is converted into a generalized eigenvalue problem[cite: 44]:

$$
A\psi_{i}=\epsilon_{i}M\psi_{i}
$$
 [cite_start][cite: 45]
-   [cite_start]$A \in \mathbb{R}^{n \times n}$: The discretized Hamiltonian matrix [cite: 46][cite_start], often represented as a sum of the kinetic energy matrix ($T$) and the diagonal potential energy matrix ($V$)[cite: 47].
-   [cite_start]$M \in \mathbb{R}^{n \times n}$: The mass matrix[cite: 48]. [cite_start]For FDM, this is typically the identity matrix[cite: 48].
-   [cite_start]$\psi_{i} \in \mathbb{R}^{n}$: The discrete eigenvector for the $i$-th orbital[cite: 49].
-   [cite_start]$\epsilon_{i}$: The eigenvalue corresponding to the $i$-th orbital's energy[cite: 50].

**Input Parameters & Physical Meaning**
-   [cite_start]$Z_{i}$: The atomic number for the $i$-th atom (e.g., Hydrogen=1, Carbon=6), which defines the nuclear charge[cite: 52].
-   [cite_start]$R_{i} \in \mathbb{R}^{3}$: The 3D coordinates of each atom, defining the molecular geometry[cite: 52].
-   [cite_start]$N$: The total number of atoms, which determines the overall size and complexity of the system[cite: 52].
-   [cite_start]Total charge / number of electrons: This determines the electron filling arrangement for the self-consistent field solution[cite: 52].

**Implementation Note**
[cite_start]The implementation involves discretizing the Schrödinger equation over a spatial grid using the Finite Difference Method (FDM)[cite: 44]. [cite_start]This transforms the differential equation into a matrix eigenvalue problem $A\psi = \epsilon M \psi$, which is then solved to find the orbital energies and wavefunctions[cite: 45].

---

### 5.3 Electromagnetic Cavity Resonance (TE Modes)

**Governing PDE (2D TE case)**  

$$
\nabla \cdot \left( \frac{1}{\mu(x,y)} \nabla E_z(x,y) \right) = -\lambda \, \epsilon(x,y) E_z(x,y),
\quad \lambda = \omega^2,
$$
 [cite_start][cite: 57, 58]
[cite_start]where \(E_z(x,y)\) is the scalar out-of-plane electric field[cite: 59].

**Matrix Formulation**  

$$
A e = \lambda M e
$$
 [cite_start][cite: 69]
- [cite_start]\(A_{ij} = \int_\Omega \frac{1}{\mu} \nabla \phi_i \cdot \nabla \phi_j \, dx\) [cite: 68]  
- [cite_start]\(M_{ij} = \int_\Omega \epsilon \, \phi_i \phi_j \, dx\) [cite: 71]  

**Input Parameters & Physical Meaning**  
- [cite_start]\(\epsilon(x,y)\): permittivity distribution (affects resonance modes) [cite: 60, 75]  
- [cite_start]\(\mu(x,y)\): permeability distribution (often constant \(\mu_0\)) [cite: 61]  

**Implementation Note**  
- [cite_start]Linear FEM discretization on rectangular mesh. [cite: 64]  
- Eigenvalue problem solved with Krylov-Schur / JD / LOBPCG solvers.  

---

### 5.4 Piezoelectric Coupled-Field Modes

**Governing Equations**  
[cite_start]Piezoelectric materials couple mechanical strain and electric displacement[cite: 77]. [cite_start]The linearized constitutive relations are[cite: 86]:

$$
\sigma_{ij} = c_{ijkl} \varepsilon_{kl} - e_{kij} E_k,
$$
 [cite_start][cite: 87]
$$
D_i = e_{ikl} \varepsilon_{kl} + \epsilon_{ij} E_j,
$$
 [cite_start][cite: 88]
where stress \(\sigma\), strain \(\varepsilon\), electric field \(E\), and displacement \(D\) interact.

**Generalized Eigenvalue Problem (after FEM discretization):**

$$
\begin{bmatrix}K_{uu}&K_{u\phi}\\ K_{\phi u}&K_{\phi\phi}\end{bmatrix}\begin{bmatrix}u\\ \phi\end{bmatrix}=\omega^{2}\begin{bmatrix}M_{uu}&0\\ 0&0\end{bmatrix}\begin{bmatrix}u\\ \phi\end{bmatrix}
$$
 [cite_start][cite: 107]
- [cite_start]\(A\): coupled stiffness–electrical matrix [cite: 101]  
- [cite_start]\(M\): mass matrix (mechanical inertia + dielectric effects) [cite: 102, 112]  
- [cite_start]\(u\): vector of mechanical + electrical degrees of freedom [cite: 107]  

**Input Parameters & Physical Meaning**  
- [cite_start]\(c_{ijkl}(x,y)\): elastic stiffness tensor [cite: 114]  
- [cite_start]\(e_{ijk}(x,y)\): piezoelectric coupling tensor [cite: 114]  
- [cite_start]\(\epsilon_{ij}(x,y)\): dielectric tensor [cite: 114]  
- [cite_start]\(\rho(x,y)\): density distribution [cite: 114]  
- [cite_start]Geometry & boundary conditions: free/constrained edges, electrodes [cite: 114]  

**Implementation Note**  
- [cite_start]Mixed FEM formulation with mechanical and electrical DOFs. [cite: 105]  
- [cite_start]Assembled into large block matrix system. [cite: 107]  

---

### 5.5 Thermal Diffusion Eigenmodes

**Governing PDE**  

$$
- \nabla \cdot \big( k(x,y) \nabla u(x,y) \big) = \lambda c(x,y) u(x,y),
$$
 [cite_start][cite: 118]
[cite_start]where \(u(x,y)\) is the temperature mode shape [cite: 119][cite_start], \(\lambda\) is the decay rate[cite: 123].

**Matrix Formulation**  

$$
K u_h = \lambda M u_h
$$
 [cite_start][cite: 129]
- [cite_start]\(K\): stiffness matrix from thermal conductivity [cite: 126]  
- [cite_start]\(M\): mass matrix from heat capacity density [cite: 127]  

**Input Parameters & Physical Meaning**  
- [cite_start]\(k(x,y)\): thermal conductivity (heat transport ability) [cite: 130]  
- [cite_start]\(c(x,y)\): volumetric heat capacity (thermal inertia) [cite: 130]  

**Implementation Note**  
- [cite_start]FEM discretization of heat operator. [cite: 125]  
- [cite_start]Eigenmodes correspond to spatial thermal decay patterns. [cite: 116]
