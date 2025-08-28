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

## 3. Dataset Generation

All datasets are generated in parallel using mpirun.
The general command format is:
```bash
mpirun -np <num_processes> python3 datasets/<script_name>.py --out <output_dir> --num <num_samples> --nx <X_dim> --ny <Y_dim> --nev <num_eigenvalues>
```

- <num_processes>: Recommended to set equal to the number of CPU cores on your machine (e.g., 4 or 8).

- <script_name>: The Python script corresponding to the dataset you want to generate.

- <output_dir>: The root directory where the generated dataset will be saved.

- <num_samples>: Total number of samples to generate.

- <X_dim> / <Y_dim>: Defines the discretized mesh size of the problem domain. The final matrix size is nx * ny.

- <num_eigenvalues>: The number of smallest eigenvalues to solve for each sample.

### 3.1 EGFR Dataset

Generate 100 samples, each with a matrix size of 50000 × 50000, solving for the 500 smallest eigenvalues:

```bash
mpirun -np 4 python3 datasets/build_egfr_parallel.py --out data/egfr_dataset --num 100 --nbas 50000 --nev 500
```

### 3.2 Electromagnetic Cavity (EM Cavity) Dataset

Generate 100 samples, each with a matrix size of 50000 × 50000, solving for the 500 smallest eigenvalues:

```bash
mpirun -np 4 python3 datasets/build_em_cavity.py --out data/em_cavity_dataset --num 100 --nx 100 --ny 100 --nev 500
```
### 3.3 Plate Vibration Dataset

Generate 100 samples, each with a matrix size of 50000 × 50000, solving for the 500 smallest eigenvalues:
```bash
mpirun -np 4 python3 datasets/build_plate.py --out data/plate_dataset --num 100 --nx 100 --ny 100 --nev 500
```

### 3.4 Thermal Diffusion Dataset

Generate 100 samples, each with a matrix size of 50000 × 50000, solving for the 500 smallest eigenvalues:
```bash
mpirun -np 4 python3 datasets/build_thermal.py --out data/thermal_dataset --num 100 --nx 100 --ny 100 --nev 500
```
### 3.5 Piezoelectric Coupled Field Dataset

```bash
mpirun -np 4 python3 datasets/build_piezo.py --out data/piezo_dataset --num 100 --nx 71 --ny 71 --nev 500 --no_solve
```
## 3. Output Structure

After successful execution, each dataset will be organized in the specified output directory with the following structure:

<output_dir>/
├── meta.json              # Metadata of the dataset
├── <global_matrix>.npz    # (Optional) Shared matrix across all samples
└── 00000/                 # First sample
    ├── A.npz              # Sample-specific matrix A
    ├── B.npz              # Sample-specific matrix B
    ├── eigvals.npy        # Computed eigenvalues (if solved)
    └── eigvecs.npy        # Computed eigenvectors (if solved)
