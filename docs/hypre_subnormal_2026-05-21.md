# Hypre subnormal-gamma investigation — progress snapshot (2026-05-21)

Where we left off. Pick up next session.

## What the problem is

When the benchmark replays PolyFEM-generated `mat_twist` matrices through Hypre under `mpirun -np 8`, Hypre's PCG bails with `Subnormal gamma value in PCG` on every matrix from simulation step ~30 onward (4–6 iter, residual stuck at ~10⁻³). The same matrices solve cleanly in PolyFEM itself.

Root cause: the row-block MPI partition I wrote in `HypreSolver.cpp` fragments the matrix graph at arbitrary row IDs. Once `mpi_size ≥ 3`, BoomerAMG's coarsening can't build a strong-enough preconditioner; `(r, M⁻¹r)` underflows; Hypre's `pcg.c:707-712` check trips.

Verified deterministic (5× identical runs), independent of MPI_Allreduce algorithm (6× identical), independent of BoomerAMG `CoarsenType` (all four subnormal).

## What I built

### Hypre — 3 runtime knobs (env var or JSON), all default to baseline behavior

| env var | values | default | effect |
|---|---|---|---|
| `POLYSOLVE_HYPRE_PARTITION` | `row_block` \| `rank_zero` \| `metis` | `row_block` | how rows are split across MPI ranks |
| `POLYSOLVE_HYPRE_PRECOND` | `boomeramg` \| `euclid` | `boomeramg` | which Hypre preconditioner PCG/GMRES uses |
| `POLYSOLVE_HYPRE_KRYLOV` | `pcg` \| `gmres` | `pcg` | outer Krylov method |

Plus a JSON-only knob:

| JSON pointer | values | default | effect |
|---|---|---|---|
| `/Hypre/max_iter` (in `libs/polysolve/linear-solver-spec.json`) | int | `5000` (raised from upstream's 1000 during the experiment, can revert) | iteration cap |

Implementation: `libs/polysolve/src/polysolve/linear/HypreSolver.{hpp,cpp}`. New `enum`s `PartitionMode { RowBlock, RankZero, Metis }`, `PreconditionerType { BoomerAMG, Euclid }`, `KrylovType { PCG, GMRES }`. Env vars read in the ctor; JSON reads in `set_parameters()`; final values used in `factorize()` and `solve()`.

### Trilinos — 1 runtime knob

| env var | values | default | effect |
|---|---|---|---|
| `POLYSOLVE_TRILINOS_KRYLOV` | `gmres` \| `cg` | `gmres` (= current `Belos::BlockGmresSolMgr`) | Krylov method; `cg` switches to `Belos::PseudoBlockCGSolMgr` |

Implementation: `libs/polysolve/src/polysolve/linear/TrilinosSolver.{hpp,cpp}`. Existing MueLu SA-AMG preconditioner is reused for both.

### METIS as a build dependency

- New recipe `libs/polysolve/cmake/recipes/metis.cmake` — uses `find_library` against the system `libmetis-dev` (Ubuntu 5.1.0) or `$CONDA_PREFIX/lib`. No CPM pull needed.
- New CMake option `POLYSOLVE_WITH_METIS` (defaults ON). Sets `POLYSOLVE_WITH_METIS` compile flag.
- METIS is only invoked when `partition_mode == Metis`. If the flag isn't set at compile time, `POLYSOLVE_HYPRE_PARTITION=metis` is auto-downgraded to `row_block` with a warning.

### Modified files (`git status` from `libs/polysolve`)

```
M  CMakeLists.txt                                # POLYSOLVE_WITH_METIS option + linkage
M  linear-solver-spec.json                       # /Hypre/max_iter raised to 5000
M  src/polysolve/linear/HypreSolver.cpp          # 3 knobs + Phase 3/6/7 paths
M  src/polysolve/linear/HypreSolver.hpp          # enums + member vars
M  src/polysolve/linear/TrilinosSolver.cpp       # krylov switch
M  src/polysolve/linear/TrilinosSolver.hpp       # KrylovType enum
?? cmake/recipes/metis.cmake                     # new METIS find recipe
```

(Plus an inline `HYPRE_GetError()` reporting block in `HypreSolver::solve()` so any future Hypre error surfaces in stderr instead of being silently dropped.)

## Key empirical results (matrix `mat_twist / 3146 / 71_1` at `tol = 1e-10`, `max_iter = 5000`)

### Hypre axis sweep at `np = 8`

| config | iter | final_res | wall | verdict |
|---|---:|---:|---:|---|
| row_block + pcg + boomeramg (default) | 5 | 1.35e-3 | 0.05 s | ❌ subn |
| metis + pcg + boomeramg | 12 | 7.59e-4 | 0.05 s | ❌ subn |
| row_block + gmres + boomeramg | 5000 | 2.10e-5 | 4.50 s | ⚠️ stuck |
| **row_block + pcg + euclid** | **2103** | **8.11e-11** | **0.36 s** | ✅ **fastest** |
| metis + gmres + boomeramg | 1521 | 9.86e-11 | 0.91 s | ✅ |
| metis + pcg + euclid | 1 | 1.31e-3 | 0.05 s | ❌ subn |
| row_block + gmres + euclid | 5000 | 6.73e-10 | 1.18 s | ⚠️ near tol |
| metis + gmres + euclid | 5000 | 1.42e-5 | 1.24 s | ⚠️ stuck |

Caveat: `row_block + pcg + euclid` is **subnormal at `np = 1`** (Euclid BJ=1 = full ILU on the whole near-singular matrix → unstable). Baseline (row_block + pcg + boomeramg) is the right choice at `np = 1`.

### Reference (same matrix, other solvers, default config)

| solver | parallelism | iter | wall | per-iter |
|---|---|---:|---:|---:|
| AMGCL (SA-AMG + CG) | OMP=8 | 127 | 1.98 s | 15.6 ms |
| Trilinos (MueLu + BlockGMRES, current default) | mpi np=8 | 527 | 0.93 s | 1.76 ms |
| Trilinos (MueLu + PseudoBlockCG) | mpi np=8 | 541 | 0.83 s | 1.53 ms |
| Trilinos (MueLu + PseudoBlockCG) | mpi np=1 | 550 | **0.67 s** | 1.22 ms |

## Three open questions for the next meeting

1. **Hypre default policy.** Two real candidates:
   - (a) `metis + gmres + boomeramg` uniform across `np` (0.91 s at `np=8`, 2.39 s at `np=1`; one mental model)
   - (b) Hybrid by `mpi_size`: baseline at `np=1`, `row_block + pcg + euclid` at `np ≥ 2` (1.10 s / 0.36 s; fastest per-`np`; ~10 lines in ctor)
   
2. **Trilinos default Krylov.** Flip to `PseudoBlockCG` (~2.6× faster at `np=1`, SPD-correct), or keep `BlockGmres` default with CG as opt-in?

3. **Meta-question:** PolyFEM force-projects every Hessian to PSD, so every benchmark matrix is SPD. Should the universal policy be *"CG by default; GMRES only when SPD doesn't hold (adjoint / mixed / raw Newton)"*? Resolving this decides Q1 + Q2 in one go.

## Artifacts produced this session

- `analysis/analysis_2026-5-21-hypre-subnormal-issue.ipynb` — full diagnosis writeup (Parts 1–4)
- `analysis/analysis_2026-5-21-hypre-subnormal-issue.pdf` — exported for Slack/sharing
- `analysis/analysis_2026-5-20-mat-twist.ipynb` — supervisor-facing summary (storage + mat_twist time/iter plots)
- `analysis/analysis_2026-4-27-explore-steps-iteration.ipynb` — per-(step, iter) timings, both repeats
- `analysis/analysis_2026-4-22_ball_rollers.ipynb` — ball_rollers comparison + extrapolation

## How to reproduce the canonical experiments

```bash
# common setup
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
BIN=/u/1/chenyang/benchmark/build.trilinos_tpetra/TestMatLogger
A=/mnt/hdd1/chenyang/benchmark_data/larger_matrix_exp/mat_twist/trial_1_result_part1/3146/71_1_A.bin
B=/mnt/hdd1/chenyang/benchmark_data/larger_matrix_exp/mat_twist/trial_1_result_part1/3146/71_1_b.bin

# default Hypre = the subnormal-failing case
mpirun -np 8 $BIN $A $B Hypre

# fastest converging Hypre config at np >= 2
POLYSOLVE_HYPRE_PRECOND=euclid \
mpirun -x POLYSOLVE_HYPRE_PRECOND -np 8 $BIN $A $B Hypre

# "principled" Hypre stack that works uniformly
POLYSOLVE_HYPRE_PARTITION=metis POLYSOLVE_HYPRE_KRYLOV=gmres \
mpirun -x POLYSOLVE_HYPRE_PARTITION -x POLYSOLVE_HYPRE_KRYLOV -np 8 $BIN $A $B Hypre

# Trilinos with CG instead of GMRES
POLYSOLVE_TRILINOS_KRYLOV=cg \
mpirun -x POLYSOLVE_TRILINOS_KRYLOV -np 8 $BIN $A $B Trilinos
```

## Next-session TODOs

1. Decide Q3 (the meta-question), which drives Q1 + Q2.
2. If hybrid Hypre default is chosen: add ~10 lines in `HypreSolver::HypreSolver()` to set `precond_type_ = Euclid` when `mpi_size_ ≥ 2` (and possibly `partition_mode_ = Metis` too, depending).
3. Revert `linear-solver-spec.json /Hypre/max_iter` back to `1000` if we don't want to leave it at `5000` for production.
4. Run the chosen default config across the full `mat_twist` dataset (and `ball_rollers`) to confirm it doesn't regress anything else; refresh the comparison plots in the corresponding analysis notebooks.
5. Optional: also expose `POLYSOLVE_HYPRE_PARTITION`/`KRYLOV`/`PRECOND` via JSON for users who don't want to set env vars (`set_parameters` already supports it, just needs JSON-schema entries in `linear-solver-spec.json`).
