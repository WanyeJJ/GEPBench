import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import sys
import os
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


def build_piezo_dataset(out_dir: str, num_samples: int, nx: int = 32, ny: int = 32,
                        seed: int = 0, grf_sigma_cells: float = 3.0, rho_mean: float = 1.0,
                        rho_amp: float = 0.1, eps_mean: float = 1.0, eps_amp: float = 0.1,
                        coupling_scale: float = 0.05, nev: int = 10, solve: bool = True):
    """Simplified piezoelectric coupled-field GEP (block matrices). Parallel version."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Rank 0 creates the root output directory and meta file
    if rank == 0:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        hx = 1.0 / (nx + 1)
        hy = 1.0 / (ny + 1)
        meta = {"problem": "Piezoelectric Coupled-Field (emulated sparse, parallel)", "nx": nx, "ny": ny, "hx": hx, "hy": hy, "nev": nev,
                "grf_sigma_cells": grf_sigma_cells, "rho_mean": rho_mean, "rho_amp": rho_amp, "eps_mean": eps_mean,
                "eps_amp": eps_amp, "coupling_scale": coupling_scale}
        with open(Path(out_dir) / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    
    # Distribute samples across processes
    samples_per_proc = num_samples // size
    start_idx = rank * samples_per_proc
    end_idx = (rank + 1) * samples_per_proc if rank < size - 1 else num_samples

    rng = np.random.default_rng(seed + rank) # Different seed for each process

    for idx in range(start_idx, end_idx):
        hx = 1.0 / (nx + 1)
        hy = 1.0 / (ny + 1)
        n = nx * ny
        L = assemble_laplacian_dirichlet(nx, ny, hx, hy)

        rho = (rho_mean + generate_grf_2d(nx, ny, grf_sigma_cells, rho_amp, int(rng.integers(1e9)))).clip(min=1e-6)
        eps = (eps_mean + generate_grf_2d(nx, ny, grf_sigma_cells, eps_amp, int(rng.integers(1e9)))).clip(min=1e-6)
        Muu = assemble_mass_diag(nx, ny, hx, hy, rho)
        Kuu = L
        Kphiphi = sp.diags(eps.reshape(-1), 0, shape=(n, n), format="csr") @ L
        Cmat = coupling_scale * sp.eye(n, format="csr")
        A = sp.bmat([[Kuu, Cmat], [Cmat, Kphiphi]], format="csr")
        B = sp.bmat([[Muu, None], [None, sp.csr_matrix((n, n))]], format="csr")

        # Each process creates its own sample directory
        sample_dir = Path(out_dir) / f"{idx:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        sp.save_npz(sample_dir / "A.npz", A)
        sp.save_npz(sample_dir / "B.npz", B)
        
        if solve:
            A_p = scipy_to_petsc_aij(A, comm=comm)
            B_p = scipy_to_petsc_aij(B, comm=comm)
            lams = solve_gep_krylovschur_parallel(A_p, B_p, nev=nev, tol=1e-8)
            
            if lams.size:
                np.save(sample_dir / "eig.npy", lams)
                print(f"[Rank {rank}] [{idx+1}/{num_samples}] saved to {sample_dir}, min/max λ: {lams.min():.6g}/{lams.max():.6g}")
            else:
                print(f"[Rank {rank}] [{idx+1}/{num_samples}] saved matrices to {sample_dir} (no convergence)")
        else:
            print(f"[Rank {rank}] [{idx+1}/{num_samples}] matrices saved to {sample_dir} (no_solve)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--num", type=int, default=100)
    p.add_argument("--nx", type=int, default=71)   # 71*71*2 ≈ 10082
    p.add_argument("--ny", type=int, default=71)
    p.add_argument("--nev", type=int, default=20)
    p.add_argument("--no_solve", action="store_true")
    args = p.parse_args()
    build_piezo_dataset(args.out, args.num, nx=args.nx, ny=args.ny, nev=args.nev, solve=not args.no_solve)
