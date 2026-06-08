# Hypre subnormal-gamma investigation — progress snapshot (2026-06-02)

Follow-up to `docs/hypre_subnormal_2026-05-21.md`. That snapshot blamed the row-block MPI partition
for "weakening" BoomerAMG. This session reframes the root cause (it is a **non-SPD preconditioner**,
not a weak one), pins it to the **smoother**, and lands a confirmed fix + best-config recommendation.

Notebooks: `analysis/analysis_2026-6-2-hypre-subnormal-issue.ipynb` (diagnosis) and
`analysis/analysis_2026-6-2-hypre-subnormal-fix.ipynb` (actionable fix + config tables).

## The reframing (what changed vs 5-21)

My mentor verified by full eigendecomposition that the `mat_twist` matrix `A` is **truly SPD**, yet
PCG hits a "negative pivot" once distributed — i.e. the **AMG operator (preconditioner) becomes
non-SPD**. This is the same event as Hypre's "Subnormal gamma":

- `pcg.c:707` bails on `if (!(gamma > HYPRE_REAL_MIN))` where `gamma = (r, M⁻¹r)`. That check fires on
  **any non-positive gamma**, and `(r, M⁻¹r) ≤ 0` is *only* possible when the preconditioner `M` is
  **not SPD**. So "subnormal gamma" == "negative pivot" == "AMG operator not SPD". Source-verified.
- So it is **not** a weak preconditioner from graph fragmentation (the 5-21 framing). Evidence already
  in 5-21 contradicted that: METIS (a better partition) does *not* fix it, and GMRES (tolerates
  non-SPD `M`) runs where PCG bails. Both are signatures of a **non-SPD `M`**, not a weak one.

## Root cause: the smoother (`relax_type`), across BOTH ranks and threads

A BoomerAMG V-cycle is SPD only if its smoother is symmetric + positive-definite. The default smoother
is **hybrid Gauss–Seidel** (`relax_type=8`, l1-scaled symmetric): GS *inside* a partition, but
couplings *across* a partition boundary are handled Jacobi-style with stale values. That drops the
cross-partition coupling from the effective smoother matrix `S` (block-diagonal local-GS), and the
more it drops, the thinner the V-cycle's SPD margin — until on a hard matrix it goes negative.

Key new finding: a "partition" is **either an MPI rank or an OpenMP thread** — both fragment the GS
smoother. The failure is governed by the **number of GS sub-domains = MPI ranks × OMP threads/rank**:

- at 1 thread/rank the cliff is **np ≥ 3** (reproduces 5-21's "np=3 cliff", and its 544/573 iters);
- at ≥16 threads/rank it bails **even at np=1**.

Subtle co-variable: with `OMP_NUM_THREADS` unset, OpenMPI binding sets the effective thread count
(`-np 1` binds to 1 core → ~1 thread → safe; `--bind-to none` → 128 threads → bails). So timing/repro
runs must pin `OMP_NUM_THREADS`.

Ruled out by experiment (issue notebook EXP-0/1/3): the matrix `A` (SPD, no subnormal entries, all
row-block diagonal blocks SPD, conditioning flat across steps), the coarse solve (Gaussian
elimination, exact), and the coarsening/interpolation/Galerkin operators (l1-Jacobi with the *same*
hierarchy is SPD).

## Exhaustive smoother test — only 2 of 22 relax_types are SPD-safe under parallelism

Tested **all 22** implemented `relax_type` values (issue notebook EXP-7) on `mat_twist/3146/71_1`,
np=4, step 71:

- ✅ converge **and** stay SPD: **16 (Chebyshev)**, **18 (l1-Jacobi)** — only these two.
- ❌ BAIL (non-SPD): 0, 3, 4, 5, 6, **8 (default)**, 7, 10, 11, 12, 17 (all hybrid-GS variants, plain
  Jacobi, FCF-Jacobi).
- ❌ never converge (non-symmetric / variable preconditioner): 1, 2, 13, 14, 15.
- ❌ not a smoother / too slow (direct solves, Kaczmarz): 9, 19, 20, 98.

Why 16/18 are immune: Chebyshev is a polynomial in `A` (only matvecs → identical for any partition);
l1-Jacobi is purely diagonal (no neighbor coupling to drop). l1-scaling — not mere diagonality — is
what matters (plain Jacobi `relax_type=0/7` bails; l1-Jacobi `18` does not).

## The fix + best config (fix notebook §2–3, `OMP_NUM_THREADS=1`, mat_twist/3146/71_1)

Baseline (`row_block + pcg + boomeramg`, dim=1): `relax=8` bails at np=8; `relax=18` → 1323 it /
0.75 s ✅; `relax=16` → 909 it / 0.99 s ✅. (l1-Jacobi beats Chebyshev on **wall time** despite more
iterations — Chebyshev costs ~2 matvecs/iter + eigenvalue-estimate setup.)

Best-config search, dim=3 + node-aligned, np=8:

| stack | iters | time |
|---|---:|---:|
| `row_block + pcg + boomeramg + hybrid-GS(8)` (baseline) | — | **BAIL** |
| **`metis + pcg + boomeramg + l1-Jacobi(18)`** (best PCG) | 562 | 0.279 s |
| `metis + pcg + boomeramg + Chebyshev(16)` | 440 | 0.407 s |
| `row_block + pcg + euclid` | 2098 | 0.358 s |
| `metis + gmres + boomeramg + hybrid-GS(8)` (formal best, = 5-21) | 256 | 0.243 s |

- Euclid only pairs with `row_block` (metis breaks its block-Jacobi ILU → bail; also bails at np=1).
- GMRES wants `relax=8` (stronger smoother, fewer iters; GMRES tolerates non-SPD `M`). Giving GMRES
  16/18 only slows it down. So the SPD-safe smoothers are needed *only* for PCG.

**Recommendation (we project to PSD, so prefer PCG):**
`metis + pcg + boomeramg`, `dim=3` + node-aligned, **`relax_type=18` (l1-Jacobi)** — fastest PCG
(0.279 s), SPD-clean, ~15 % slower than the formal-best GMRES stack. Use `16` (Chebyshev) if
optimizing iteration count/memory. `metis + gmres + boomeramg + relax=8` remains the fastest overall
and the fallback for non-SPD systems.

## Code / artifacts this session

- Added 2 diagnostic env toggles to `libs/polysolve/src/polysolve/linear/HypreSolver.cpp` (branch
  `fix_subnormal_hypre`), inside `HypreBoomerAMG_SetDefaultOptions`, defaults unchanged when unset:
  - `POLYSOLVE_HYPRE_RELAX` — overrides smoother type (e.g. 8/16/18).
  - `POLYSOLVE_HYPRE_MAXLEVELS` — caps hierarchy depth (=1 → pure smoother, used to isolate smoother
    from hierarchy in EXP-3).
  - Rebuilt `build.trilinos_tpetra/`. Toggles can stay as diagnostics or be reverted.
- Scripts under `analysis/`: `subnormal_matrix_probe.py` (EXP-0), `subnormal_hypre_experiments.sh`
  (EXP-1–4), `subnormal_confirm_fix.sh` (EXP-5), `subnormal_np_omp_map.sh` (EXP-6),
  `subnormal_all_relax.sh` (EXP-7), `subnormal_fix_configs.sh` (fix §2–3), plus the notebook builders
  `build_subnormal_notebook.py` / `build_fix_notebook.py`.

## Next / open

- Decide whether to make `relax_type=18` the automatic PCG default at np≥2 in `HypreSolver.cpp` (a few
  lines in the ctor switching on `mpi_size_`), or keep it as an env opt-in + documentation.
- The benchmark currently runs Hypre with `OMP_NUM_THREADS` unset; for fair/correct PCG timing it
  should pin threads (the thread count is itself a co-trigger).
