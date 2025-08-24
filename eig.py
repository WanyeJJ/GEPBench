import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc
from slepc4py import SLEPc


def scipy_to_petsc_aij(A: sp.csr_matrix, comm=PETSc.COMM_WORLD) -> PETSc.Mat:
    """
    Converts a global SciPy CSR matrix to a parallel PETSc AIJ matrix.
    Each process is responsible for inserting its local portion of the matrix.
    This is the parallel-safe version.
    """
    # Create an empty distributed PETSc matrix
    # The size argument is the GLOBAL size of the matrix.
    mat = PETSc.Mat().createAIJ(size=A.shape, comm=comm)
    mat.setUp()

    # Get the local ownership range for the rows
    rstart, rend = mat.getOwnershipRange()
    
    # Extract CSR components
    indptr = A.indptr
    indices = A.indices
    data = A.data

    # Each process sets the values for the rows it owns
    for row in range(rstart, rend):
        row_start_idx = indptr[row]
        row_end_idx = indptr[row + 1]
        
        cols = indices[row_start_idx:row_end_idx].astype(PETSc.IntType)
        vals = data[row_start_idx:row_end_idx].astype(np.float64)
        
        if cols.size > 0:
            mat.setValues(row, cols, vals)

    mat.assemble()
    return mat


def solve_gep_krylovschur(A: PETSc.Mat, B: PETSc.Mat, nev: int, tol: float = 1e-12, maxit: int = 1000) -> np.ndarray:
    """Serial Krylov-Schur solver."""
    eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
    eps.setOperators(A, B)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setTolerances(tol=tol, max_it=maxit)
    
    mpd = max(2 * int(nev), 50)
    ncv = min(4 * int(nev), int(nev) + mpd - 1)
    ncv = max(ncv, int(nev) + 1)
    
    eps.setDimensions(nev=nev, ncv=ncv, mpd=mpd)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
    eps.setFromOptions()
    eps.solve()
    nconv = eps.getConverged()
    vals = []
    for i in range(min(nconv, int(nev))):
        lam = eps.getEigenpair(i)[0]
        vals.append(float(lam))
    vals = np.array(sorted(vals)) if len(vals) else np.array([])
    eps.destroy()
    return vals


def solve_gep_krylovschur_fast(A: PETSc.Mat, B: PETSc.Mat, nev: int, tol: float = 1e-8, maxit: int = 500) -> np.ndarray:
    """Fast version of Krylov-Schur solver with relaxed tolerance and optimized parameters."""
    eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
    eps.setOperators(A, B)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setTolerances(tol=tol, max_it=maxit)
    
    mpd = min(4 * int(nev), 300)
    ncv = min(3 * int(nev), int(nev) + mpd - 1)
    ncv = max(ncv, int(nev) + 1)
    
    eps.setDimensions(nev=nev, ncv=ncv, mpd=mpd)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
    eps.setFromOptions()
    eps.solve()
    nconv = eps.getConverged()
    vals = []
    for i in range(min(nconv, int(nev))):
        lam = eps.getEigenpair(i)[0]
        vals.append(float(lam))
    vals = np.array(sorted(vals)) if len(vals) else np.array([])
    eps.destroy()
    return vals


def solve_gep_lanczos(A: PETSc.Mat, B: PETSc.Mat, nev: int, tol: float = 1e-8, maxit: int = 500) -> np.ndarray:
    """Lanczos solver, often faster for symmetric problems."""
    eps = SLEPc.EPS().create(PETSc.COMM_WORLD)
    eps.setOperators(A, B)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setType(SLEPc.EPS.Type.LANCZOS)
    eps.setTolerances(tol=tol, max_it=maxit)
    
    mpd = min(3 * int(nev), 200)
    ncv = min(2 * int(nev), int(nev) + mpd - 1)
    ncv = max(ncv, int(nev) + 1)
    
    eps.setDimensions(nev=nev, ncv=ncv, mpd=mpd)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
    eps.setFromOptions()
    eps.solve()
    nconv = eps.getConverged()
    vals = []
    for i in range(min(nconv, int(nev))):
        lam = eps.getEigenpair(i)[0]
        vals.append(float(lam))
    vals = np.array(sorted(vals)) if len(vals) else np.array([])
    eps.destroy()
    return vals


def solve_gep_krylovschur_parallel(A: PETSc.Mat, B: PETSc.Mat, nev: int, tol: float = 1e-8, maxit: int = 500) -> np.ndarray:
    """Parallel version of Krylov-Schur solver with corrected solver configuration."""
    comm = A.getComm()
    eps = SLEPc.EPS().create(comm=comm)
    eps.setOperators(A, B)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setTolerances(tol=tol, max_it=maxit)
    
    mpd = min(2 * int(nev), 150)
    ncv = min(3 * int(nev), int(nev) + mpd - 1)
    ncv = max(ncv, int(nev) + 1)
    
    eps.setDimensions(nev=nev, ncv=ncv, mpd=mpd)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
    
    # --- THIS IS THE FINAL FIX FOR THE PARALLEL SOLVER ---
    # Get the solver objects
    st = eps.getST()
    ksp = st.getKSP()
    
    # Set the main preconditioner type
    pc = ksp.getPC()
    pc.setType('bjacobi')  # Use Block Jacobi as the parallel preconditioner

    # Use the PETSc Options database to configure the sub-solvers
    # This is the correct way to set options for nested objects before they are set up.
    opts = PETSc.Options()
    prefix = ksp.getOptionsPrefix()
    opts[f"{prefix}sub_pc_type"] = "ilu"
    
    # Apply these options and any others from the command line
    ksp.setFromOptions()
    # --- END OF FIX ---
    
    eps.solve()
    
    nconv = eps.getConverged()
    vals = []
    for i in range(min(nconv, int(nev))):
        # --- FIX FOR TypeError ---
        # Safely get the eigenvalue, whether it's a tuple or a float
        eigenpair = eps.getEigenpair(i)
        if isinstance(eigenpair, tuple):
            lam = eigenpair[0]
        else:
            lam = eigenpair
        # --- END OF FIX ---
        vals.append(float(lam))
        
    vals = np.array(sorted(vals)) if len(vals) else np.array([])
    eps.destroy()
    return vals