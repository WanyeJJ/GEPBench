import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import sys
import os
from mpi4py import MPI

# Add the datasets directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    scipy_to_petsc_aij,
    solve_gep_krylovschur_parallel,
)


def laplacian_1d(n: int) -> sp.csr_matrix:
    """Generates a 1D Laplacian matrix."""
    main = 2.0 * np.ones(n)
    off = -1.0 * np.ones(n - 1)
    A = sp.diags([off, main, off], [-1, 0, 1], shape=(n, n), format="csr")
    return A


def build_egfr_dataset_parallel(out_dir: str, num_samples: int, nbas: int = 128,
                                seed: int = 0, nev: int = 500, solve: bool = True):
    """EGFR electronic structure GEP H c = λ S c (emulated, sparse) with parallel solving."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Rank 0 is responsible for creating the main output directory and meta file
    if rank == 0:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        meta = {"problem": "EGFR Electronic Structure (emulated sparse, parallel)", "nbas": nbas, "nev": nev}
        with open(Path(out_dir) / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    
    # Distribute samples across all available processes
    samples_per_proc = num_samples // size
    start_idx = rank * samples_per_proc
    # Ensure the last process handles any remaining samples
    end_idx = (rank + 1) * samples_per_proc if rank < size - 1 else num_samples
    
    rng = np.random.default_rng(seed + rank)  # Use a different seed for each process to ensure variety
    
    # Each process iterates through its assigned range of samples
    for idx in range(start_idx, end_idx):
        T = laplacian_1d(nbas)
        V = sp.diags(rng.uniform(-1.0, 1.0, size=nbas), 0, format="csr")
        H = T + V
        S = sp.diags(rng.uniform(0.5, 1.5, size=nbas), 0, format="csr")
        
        # --- FIX: Each process creates the directory for the sample it is responsible for ---
        sample_dir = Path(out_dir) / f"{idx:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # The process that generated the data is responsible for saving it
        sp.save_npz(sample_dir / "H.npz", H)
        sp.save_npz(sample_dir / "S.npz", S)
        
        if solve:
            A_p = scipy_to_petsc_aij(H, comm=comm)
            B_p = scipy_to_petsc_aij(S, comm=comm)
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
    p.add_argument("--num", type=int, default=10)
    p.add_argument("--nbas", type=int, default=128)
    p.add_argument("--nev", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_solve", action="store_true")
    args = p.parse_args()
    build_egfr_dataset_parallel(args.out, args.num, nbas=args.nbas, seed=args.seed, nev=args.nev, solve=not args.no_solve)
