import os
import json
import numpy as np
import scipy.sparse as sp
from pathlib import Path
import sys
from mpi4py import MPI

# Add the datasets directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# FIX: Changed from relative to absolute import
from utils import (
    generate_grf_2d,
    assemble_laplacian_dirichlet,
    assemble_mass_diag,
    scipy_to_petsc_aij,
    solve_gep_krylovschur_parallel, # Use the parallel solver
)


def build_plate_dataset(out_dir: str, num_samples: int, nx: int = 64, ny: int = 64,
                        seed: int = 0, grf_sigma_cells: float = 3.0, density_mean: float = 1.0,
                        density_amp: float = 0.2, nev: int = 10, solve: bool = True):
    """Kirchhoff-Love plate vibration GEP: Au = λ M u with GRF density ρ(x). Parallel version."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Rank 0 is responsible for creating the main output directory and meta file
    if rank == 0:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        hx = 1.0 / (nx + 1)
        hy = 1.0 / (ny + 1)
        meta = {"problem": "Kirchhoff-Love Plate (emulated sparse, parallel)", "nx": nx, "ny": ny, "hx": hx, "hy": hy, "nev": nev,
                "grf_sigma_cells": grf_sigma_cells, "density_mean": density_mean, "density_amp": density_amp}
        with open(Path(out_dir) / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    # All processes build the constant K matrix.
    hx = 1.0 / (nx + 1)
    hy = 1.0 / (ny + 1)
    K = assemble_laplacian_dirichlet(nx, ny, hx, hy)
    
    # Only Rank 0 saves the global K matrix to disk once.
    if rank == 0:
        sp.save_npz(Path(out_dir) / "K.npz", K)
    
    # Distribute samples across processes
    samples_per_proc = num_samples // size
    start_idx = rank * samples_per_proc
    end_idx = (rank + 1) * samples_per_proc if rank < size - 1 else num_samples

    rng = np.random.default_rng(seed + rank) # Different seed for each process

    for idx in range(start_idx, end_idx):
        grf = generate_grf_2d(nx, ny, sigma_cells=grf_sigma_cells, amplitude=density_amp, seed=int(rng.integers(1e9)))
        rho = (density_mean + grf).clip(min=1e-3)
        M = assemble_mass_diag(nx, ny, hx, hy, rho)

        # Each process creates its own sample directory
        sample_dir = Path(out_dir) / f"{idx:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        sp.save_npz(sample_dir / "M.npz", M)
        
        if solve:
            # We convert the full K and M matrices to distributed PETSc matrices
            A_petsc = scipy_to_petsc_aij(K, comm=comm)
            B_petsc = scipy_to_petsc_aij(M, comm=comm)
            lams = solve_gep_krylovschur_parallel(A_petsc, B_petsc, nev=nev, tol=1e-8)
            
            if lams.size:
                np.save(sample_dir / "eig.npy", lams)
                print(f"[Rank {rank}] [{idx+1}/{num_samples}] saved to {sample_dir}, min/max λ: {lams.min():.6g}/{lams.max():.6g}")
            else:
                print(f"[Rank {rank}] [{idx+1}/{num_samples}] saved matrices to {sample_dir} (no convergence)")
        else:
            print(f"[Rank {rank}] [{idx+1}/{num_samples}] matrices saved to {sample_dir} (no_solve)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--nx", type=int, default=100)  # nx*ny = 10000
    parser.add_argument("--ny", type=int, default=100)  # Total size = 10000x10000
    parser.add_argument("--nev", type=int, default=20)
    parser.add_argument("--no_solve", action="store_true")
    args = parser.parse_args()
    build_plate_dataset(args.out, args.num, nx=args.nx, ny=args.ny, nev=args.nev, solve=not args.no_solve)
