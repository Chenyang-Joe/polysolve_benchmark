# Trilinos Solver Design Decisions

## Overview

TrilinosSolver integrates Trilinos (Belos + MueLu) into polysolve as a distributed linear solver for elasticity problems. This document records parameter choices, benchmark results, and design rationale.

**Final configuration:**
- Krylov solver: **GMRES** (Belos BlockGmresSolMgr)
- Preconditioner: **MueLu** (Smoothed Aggregation AMG)
- Smoother: **Chebyshev**
- Aggregation: **Uncoupled** (MueLu default)
- DOF-aligned row partitioning for elasticity

---

## 1. Krylov Solver: Why GMRES Instead of CG

### Background

Hypre uses **PCG** (Preconditioned Conjugate Gradient) for SPD elasticity problems. CG is mathematically optimal for SPD systems — it minimizes the A-norm of the error over the Krylov subspace and has lower per-iteration cost than GMRES (no orthogonalization or restart overhead).

Naturally, we first tried CG (Belos PseudoBlockCGSolMgr) for Trilinos.

### Test Results

Test matrix: 3D golf ball, 339975 DOFs (113325 nodes, numPDEs=3), 14.4M nonzeros.

| Solver | Smoother | np | Iterations | Solve Time | Result |
|--------|----------|----|------------|------------|--------|
| CG | Chebyshev | 1 | 20 | - | Converged |
| CG | Chebyshev | 2 | 24 | 3.70s | Converged |
| **CG** | **Chebyshev** | **4** | **-** | **-** | **CRASH: non-positive p^H*A*p** |
| GMRES | Sym. GS | 4 | 1000 | 577s | Did not converge |
| GMRES | Chebyshev | 2 | 24 | 3.85s | Converged |
| **GMRES** | **Chebyshev** | **4** | **56** | **9.35s** | **Converged** |

### Root Cause: MueLu V-cycle Is Not SPD in Parallel

CG requires the preconditioner M^{-1} to be symmetric positive definite (SPD). The MueLu AMG V-cycle loses its SPD property when distributed across multiple processes. The specific reason is the **UncoupledAggregation** strategy:

- Each MPI rank aggregates its local rows **independently**, without cross-process coordination
- More processes = more partition boundaries = more fragmented aggregates = poorer coarse grid quality
- The coarse grid correction P * A_c^{-1} * P^T can "over-correct" in certain directions, making v^T M^{-1} v < 0 for some vectors

Evidence from MueLu verbose output (np=2 vs np=4):

| | np=2 | np=4 |
|---|---|---|
| Level 1 rows | 25,782 | 46,362 |
| Level 1 nnz | 4.6M | 10.6M |
| Coarsening ratio | 13.2x | 7.3x |
| Operator complexity | 1.32 | 1.78 |

np=4 produces a Level 1 that is almost **twice as large** as np=2, with much worse coarsening ratio, confirming the aggregation quality degradation.

### Why Hypre Can Use CG

Hypre's BoomerAMG uses a fundamentally different AMG approach designed for parallel SPD preservation:

| Component | Hypre (BoomerAMG) | MueLu (SA-AMG) |
|-----------|-------------------|----------------|
| Coarsening | **HMIS** — global C/F splitting with cross-process coordination via MPI | **UncoupledAggregation** — each process aggregates independently |
| Interpolation | **Extended+i** — uses neighbor-of-neighbor info; **GM-2** variant for elasticity | **Smoothed Aggregation** — tent function + Jacobi smoothing |
| Smoother | **L1-Gauss-Seidel** — L1 weighting at process boundaries guarantees parallel symmetry | Symmetric GS (not truly symmetric in parallel) or Chebyshev |

Hypre's entire AMG pipeline is engineered for parallel SPD preservation. MueLu's SA-AMG is a different mathematical framework (aggregation-based vs classical) that does not guarantee this property.

### Feasibility of Making MueLu SPD-Compatible

| Fix | Feasibility | Notes |
|-----|-------------|-------|
| L1-Symmetric Gauss-Seidel smoother | Possible (Ifpack2 supports `relaxation: use l1`) | Alone not sufficient — aggregation is the main issue |
| Coupled aggregation | **Not implemented** in MueLu — parameter validates but no factory handler exists | Would require Trilinos source modification |
| Classical AMG coarsening | Available but tested poorly (176 iters vs 24) | Not suitable for elasticity with SA |
| Extended+i / GM-2 interpolation | **Not available** in MueLu — architectural difference from classical AMG | Fundamental SA vs classical AMG gap |

**Conclusion:** Making MueLu work with CG on arbitrary process counts would require changes to the Trilinos codebase. GMRES is the correct choice for MueLu.

### GMRES Performance Overhead

GMRES vs CG on np=2 (where both work):
- CG: 24 iters, 3.70s solve
- GMRES: 24 iters, 3.85s solve

The overhead is minimal (~4%) because the iteration count is the same — the preconditioner quality dominates, not the Krylov method overhead.

---

## 2. Smoother: Why Chebyshev

### Options Tested

| Smoother | np=2 iters | np=4 result |
|----------|------------|-------------|
| **Symmetric Gauss-Seidel** (MueLu default) | 23 | 1000 iters, did not converge |
| **Chebyshev** | 24 | 56 iters, converged |

### Rationale

- MueLu's default smoother for generic problems is Symmetric Gauss-Seidel (Symmetric Gauss-Seidel)
- In parallel, Symmetric Gauss-Seidel is applied locally per process — the global smoother is **not truly symmetric**, which degrades GMRES convergence on multiple processes
- **Chebyshev** is a polynomial smoother that is inherently parallel-safe and symmetric regardless of process count
- MueLu's own `Elasticity-3D` preset uses Chebyshev
- Chebyshev is also the standard recommendation for SA-AMG in the literature

---

## 3. Aggregation Strategy

### Uncoupled (Default) vs Classical

| Aggregation | np=2 iters | np=2 factorize | np=2 total | np=4 iters | np=4 total |
|-------------|------------|----------------|------------|------------|------------|
| **SA Uncoupled** | **24** | 4.36s | **8.06s** | **56** | **15.4s** |
| Classical | 176 | 0.99s | 9.23s | 176 | 28.1s |

Classical aggregation builds the AMG hierarchy faster but produces a much weaker preconditioner (7x more iterations). SA Uncoupled is clearly superior for elasticity.

---

## 4. Drop Tolerance (Strength Threshold)

### SA Uncoupled with Different Drop Tolerances

Hypre uses `theta=0.5` for elasticity. We tested whether MueLu benefits from similar filtering.

| Drop Tol | np=2 iters | np=2 factorize | np=2 total | np=4 iters | np=4 total |
|----------|------------|----------------|------------|------------|------------|
| **Default (0.0)** | **24** | 4.36s | **8.06s** | **56** | **15.4s** |
| 0.5 | 117 | 1.59s | 9.64s | 118 | 20.8s |

Drop tol=0.5 reduces factorize time (fewer connections to process) but severely degrades preconditioner quality (5x more iterations). MueLu's SA approach benefits from keeping all connections — the aggregation and prolongation smoothing handle the weak connections effectively.

**This is a fundamental difference from Hypre's classical AMG**, where filtering weak connections improves C/F splitting quality.

---

## 5. DOF-Aligned Row Partitioning

### Problem

When `numPDEs > 1`, MueLu groups rows into blocks of `numPDEs` for aggregation. Tpetra's default map distributes rows evenly without considering block structure:

```
Example: 15 rows, numPDEs=3, np=2
Default map:  rank 0 gets 8 rows, rank 1 gets 7 rows
              → node 2 (rows 6,7,8) is SPLIT across ranks
              → MueLu aggregation fails or produces garbage
```

### Solution

Create a custom map that partitions by nodes first, then expands to DOFs:

```
Aligned map:  rank 0 gets 9 rows (nodes 0-2), rank 1 gets 6 rows (nodes 3-4)
              → every rank has complete nodes
              → each rank's row count is a multiple of numPDEs
```

### Impact

Without DOF alignment, np=2 on the 339975-row matrix **crashed** with heap corruption in `TentativePFactory::BuildPuncoupled`. With alignment, it converges in 24 iterations.

If `numGlobalRows % numPDEs != 0`, the solver throws an exception — this indicates a mismatch between `block_size` and the actual problem.

---

## 6. Complete Parameter Summary

### MueLu Preconditioner

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `smoother: type` | `"CHEBYSHEV"` | Parallel-safe, symmetric, optimal for SA-AMG |
| `number of equations` | `numPDEs` (from `block_size`, default 3) | Enables block aggregation for elasticity |
| `aggregation: type` | uncoupled (default) | Best iteration count for SA-AMG |
| `aggregation: drop tol` | 0.0 (default) | SA benefits from keeping all connections |
| `verbosity` | `"none"` | Production setting |
| `max levels` | 25 (default) | Sufficient for tested problem sizes |
| `coarse: max size` | 2000 (default) | Direct solve (KLU) on coarsest level |
| `multigrid algorithm` | sa (default) | Smoothed Aggregation |

### Belos Krylov Solver

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Solver type | `BlockGmresSolMgr` | MueLu V-cycle not SPD in parallel; GMRES does not require SPD preconditioner |
| `Maximum Iterations` | 1000 (configurable) | From `linear-solver-spec.json` |
| `Convergence Tolerance` | 1e-10 (configurable) | From `linear-solver-spec.json` |
| `Verbosity` | Errors + Warnings | Production setting |

### User-Configurable Parameters (via JSON)

| JSON Path | Default | Description |
|-----------|---------|-------------|
| `Trilinos/block_size` | 3 | DOFs per node (1=scalar, 2=2D, 3=3D elasticity) |
| `Trilinos/max_iter` | 1000 | Maximum Krylov iterations |
| `Trilinos/tolerance` | 1e-10 | Convergence tolerance |
| `Trilinos/is_nullspace` | true | Nullspace flag (not yet fully implemented) |

---

## 7. Benchmark Summary

Test matrix: 3D golf ball elasticity, 339975 DOFs, 14.4M nnz.

### Final Configuration (GMRES + Chebyshev + SA Uncoupled)

| np | Iterations | Factorize | Solve | Total | Converged |
|----|------------|-----------|-------|-------|-----------|
| 1  | 20 | - | - | - | Yes |
| 2  | 24 | 4.36s | 3.70s | 8.06s | Yes |
| 4  | 56 | 6.01s | 9.35s | 15.4s | Yes |

### Notes on Scaling

- np=2 → np=4: iterations increase from 24 to 56 (2.3x) due to UncoupledAggregation quality degradation
- Factorize time increases with np due to MueLu's AMG setup communication overhead
- For this problem size, np=2 gives the best time-to-solution

---

## 8. Future Improvements

1. **Nullspace vectors**: Provide full 6-vector near-nullspace (3 translations + 3 rotations) to MueLu. Currently only 3 translation vectors are auto-generated. Requires passing node coordinates to the solver.

2. **Solver selection**: Could auto-select CG for np=1 (guaranteed SPD) and GMRES for np>1.

3. **Repartitioning**: The `enable_repartition` parameter exists but is not yet implemented. MueLu supports graph-based repartitioning that could improve load balance for irregular meshes.
