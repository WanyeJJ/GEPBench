import os
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter
import sys

# Add the datasets directory to the path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# These imports are now handled directly in the new scipy_to_petsc_aij function
# from eig import scipy_to_petsc_aij as _conv

import petsc4py
from petsc4py import PETSc

def generate_grf_2d(nx: int, ny: int, sigma_cells: float, amplitude: float, seed: int | None = None) -> np.ndarray:
    """Generate a 2D Gaussian random field using Gaussian filtering.
    - sigma_cells: smoothing std in grid cells
    - amplitude: std dev after smoothing (approximately)
    """
    rng = np.random.default_rng(seed)
    field = rng.standard_normal((ny, nx)).astype(np.float64)
    field = gaussian_filter(field, sigma=sigma_cells, mode="reflect")
    std = field.std() if field.std() > 0 else 1.0
    field = field / std * amplitude
    return field


def assemble_laplacian_dirichlet(nx: int, ny: int, hx: float, hy: float) -> sp.csr_matrix:
    """Assemble 2D 5-point Laplacian with Dirichlet BC on unit square interior grid (nx by ny interior nodes)."""
    n = nx * ny
    main = np.full(n, 2.0 / hx**2 + 2.0 / hy**2)
    offx = np.full(n - 1, -1.0 / hx**2)
    offy = np.full(n - nx, -1.0 / hy**2)
    for i in range(1, ny):
        offx[i * nx - 1] = 0.0
    diags = [main, offx, offx, offy, offy]
    offsets = [0, -1, 1, -nx, nx]
    K = sp.diags(diagonals=diags, offsets=offsets, shape=(n, n), format="csr")
    return K


def assemble_mass_diag(nx: int, ny: int, hx: float, hy: float, density: np.ndarray | float) -> sp.csr_matrix:
    """Assemble diagonal mass/capacity matrix with (density * hx * hy) on diagonal."""
    if np.isscalar(density):
        vals = np.full(nx * ny, float(density) * hx * hy)
    else:
        assert density.shape == (ny, nx)
        vals = (density.reshape(-1) * hx * hy).astype(np.float64)
    M = sp.diags(vals, 0, format="csr")
    return M


# The core fix is in this function to handle parallel data insertion
def scipy_to_petsc_aij(A: sp.csr_matrix, comm=PETSc.COMM_WORLD) -> PETSc.Mat:
    """
    Converts a global SciPy CSR matrix to a parallel PETSc AIJ matrix.
    Each process is responsible for inserting its local portion of the matrix.
    """
    # Create an empty distributed PETSc matrix
    # The size argument is the GLOBAL size of the matrix.
    mat = PETSc.Mat().createAIJ(size=A.shape, comm=comm)
    mat.setUp()

    # Get the local ownership range for the rows
    rstart, rend = mat.getOwnershipRange()
    
    # Iterate through the rows of the global matrix
    # and set values only for the rows owned by the current process.
    
    indptr = A.indptr
    indices = A.indices
    data = A.data

    for row in range(rstart, rend):
        row_start_idx = indptr[row]
        row_end_idx = indptr[row + 1]
        
        cols = indices[row_start_idx:row_end_idx]
        vals = data[row_start_idx:row_end_idx]
        
        if cols.size > 0:
            mat.setValues(row, cols, vals)

    mat.assemble()
    return mat


# The lazy wrappers are unchanged
def solve_gep_krylovschur(A, B, nev: int, tol: float = 1e-12, maxit: int = 1000):
    from eig import solve_gep_krylovschur as _solve
    return _solve(A, B, nev, tol, maxit)


def solve_gep_krylovschur_parallel(A, B, nev: int, tol: float = 1e-8, maxit: int = 500):
    from eig import solve_gep_krylovschur_parallel as _solve
    return _solve(A, B, nev, tol, maxit)


def solve_gep_krylovschur_fast(A, B, nev: int, tol: float = 1e-8, maxit: int = 500):
    from eig import solve_gep_krylovschur_fast as _solve
    return _solve(A, B, nev, tol, maxit)