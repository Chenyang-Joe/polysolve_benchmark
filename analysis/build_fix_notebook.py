#!/usr/bin/env python
"""Builds analysis/analysis_2026-6-2-hypre-subnormal-fix.ipynb.
Run with: ~/miniconda3/envs/benchmark/bin/python analysis/build_fix_notebook.py"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t.strip("\n")))
def code(t, out=None):
    c = nbf.v4.new_code_cell(t.strip("\n"))
    if out is not None:
        c["outputs"] = [nbf.v4.new_output("stream", name="stdout", text=out)]
    cells.append(c)

# ===================================================================== title
md(r"""
# Hypre subnormal/non-SPD — the fix: choosing an SPD-safe smoother

*2026-06-02*

Companion to `analysis_2026-6-2-hypre-subnormal-issue.ipynb` (the diagnosis). This notebook is the
**actionable** version: what the bug is in one paragraph, which `relax_type` (smoother) to use, and
the best Hypre configuration for our PCG-based benchmark.
""")

# ===================================================================== Sec 0 prereq
md(r"""
## 0. Prerequisite — AMG V-cycle is the *preconditioner*, CG is the *solver*

We solve $A x = b$ with $A$ SPD (symmetric, all eigenvalues > 0 — a 3D-elasticity stiffness matrix).

**Two separate pieces, don't conflate them:**

```
   CG  (the SOLVER, outer iteration)
    └── calls a PRECONDITIONER once per step:  z = M⁻¹ r
                                                   └── = ONE BoomerAMG V-cycle
```

- **CG (Conjugate Gradient)** is the actual solver. It iterates until $\|b-Ax\|$ is small. CG is the
  method of choice for SPD systems but has a hard requirement: the operator it works on must be SPD.
- **A preconditioner $M$** accelerates CG. Each CG step needs to apply $M^{-1}$ to a vector
  ($z = M^{-1} r$). $M \approx A$, cheap to apply. **Preconditioned CG additionally requires $M$ to
  be SPD** — if $M$ is indefinite, CG breaks. $M$ is never built as a matrix; we only ever apply it.
- **The AMG V-cycle is that preconditioner.** "Applying $M^{-1}$" = running exactly **one** BoomerAMG
  V-cycle on the input vector.

**V-cycle as preconditioner, NOT as solver — this is the key framing.** BoomerAMG *can* be used
standalone (iterate many V-cycles until convergence). Here it is **not**: it runs **one** V-cycle per
CG step, as the preconditioner. In the code this is pinned by:

```cpp
HYPRE_BoomerAMGSetMaxIter(amg_precond, 1);   // exactly one V-cycle
HYPRE_BoomerAMGSetTol(amg_precond, 0.0);     // don't iterate to convergence
HYPRE_PCGSetPrecond(solver, HYPRE_BoomerAMGSolve, HYPRE_BoomerAMGSetup, precond);  // hand it to CG
```

**What one V-cycle is made of** (so the later vocabulary is clear):

```
V-cycle(r):  smooth (a few cheap sweeps)            ┐ "smoother" = relax_type
             → restrict residual to a coarser grid   │
             → (recurse on coarse grid)              │ hierarchy: coarsen / interpolate / Galerkin PᵀAP
             → interpolate correction back           │
             → smooth again                          ┘
             coarsest grid: solved exactly (Gaussian elimination)
```

The **smoother** (`relax_type`) is the one knob this whole investigation turns on. Why it controls
SPD-ness is Section 1.
""")

# ===================================================================== Sec 0.1 hybrid GS
md(r"""
### 0.1 What "hybrid Gauss–Seidel" actually does across partitions

The default smoother is **hybrid Gauss–Seidel**. The name is "hybrid" because it is **Gauss–Seidel
*inside* each partition, but Jacobi *between* partitions** (a partition = the rows owned by one MPI
rank or one OpenMP thread). It is *not* "Jacobi first, then a separate GS pass on the coupling" —
there is no separate coupling pass; the cross-partition coupling is handled *with stale values inside
the same local sweep*. Here is the mechanism.

**Each unknown's neighbors split into two kinds.** The GS update of row $i$ is
$x_i \leftarrow \frac{1}{a_{ii}}\big(b_i - \sum_{j\neq i} a_{ij} x_j\big)$. For each neighbor $j$
(a nonzero $a_{ij}$) we must decide: use the **new** $x_j$ (already updated this sweep) or the **old**
$x_j$ (from last sweep)?

- **neighbor in the same partition** → use the **new** value ⇒ this is true Gauss–Seidel.
- **neighbor in another partition** → use the **old** value ⇒ this is the Jacobi part.

Cross-partition neighbors *must* use old values: you cannot fetch another rank/thread's
freshly-updated value mid-sweep without massive communication that would re-serialize the work. So:

```
one hybrid-GS sweep:
  ① halo exchange: each partition receives a SNAPSHOT (old values) of its neighbors' boundary x
  ② each partition sweeps its own rows independently, in parallel:
       - local neighbor      → use the just-updated value   (Gauss–Seidel)
       - cross-partition nbr → use the frozen ① snapshot     (Jacobi / lagged)
  ③ exchange updated boundary values for the next sweep
```

Step ② runs on all partitions **independently and in parallel** — that is the whole point — at the
price that cross-partition couplings are always "one sweep behind."

**Tiny example (2 partitions).** 6 unknowns, P0={1,2,3}, P1={4,5,6}, boundary coupling 3↔4.
Updating $x_3$ needs $x_4$, which lives on P1 → use $x_4^{\text{old}}$; simultaneously P1 updates
$x_4$ using $x_3^{\text{old}}$. So the boundary pair 3↔4 is **Jacobi** (each uses the other's old
value), while P0's interior 1→2→3 is true **GS** (uses fresh values).

**Matrix view (ties to the $S$ in §0).** Block the matrix by partition,
$A=\begin{bmatrix}A_{00}&A_{01}\\A_{10}&A_{11}\end{bmatrix}$ ($A_{pp}$ = within-partition,
$A_{pq}$ = cross-partition coupling). The hybrid-GS smoother is

$$S=\begin{bmatrix}\mathrm{tril}(A_{00})&0\\ 0&\mathrm{tril}(A_{11})\end{bmatrix}$$

— **block-diagonal**, each block being its own lower-triangular GS, and the **cross blocks
$A_{01},A_{10}$ are dropped from $S$** (moved to the right-hand side as lagged/old values). With one
partition $S=\mathrm{tril}(A)$ — clean GS; the more partitions, the more coupling is dropped and the
more $S$ degrades toward pure Jacobi. (Symmetric hybrid GS, `relax_type` 6/8, just replaces each
block's `tril` by a forward-then-backward SSOR sweep; cross blocks are still lagged.)

**That dropped cross-coupling is what costs SPD** (Section 1): true GS uses the full triangular
$\mathrm{tril}(A)$, whose forward+backward pair assembles into a symmetric SSOR; hybrid GS instead
discards the cross blocks, and the more it discards (≳3 partitions) the thinner the preconditioner's
SPD margin gets — until on a hard matrix it goes negative. **l1-Jacobi** (no local-vs-cross
distinction — purely diagonal) and **Chebyshev** (only matvecs — identical for any partitioning) have
no cross-coupling to drop, which is why they are immune.
""")

# ===================================================================== Sec 1 summary
md(r"""
## 1. The bug in one paragraph

Hypre PCG aborts on `mat_twist` with *"Subnormal gamma value in PCG"* once the problem is spread
across several MPI ranks (or OpenMP threads). The diagnosis notebook works through it in detail; the
essence is simple:

> **The matrix `A` is SPD, but the *preconditioner* (the AMG V-cycle) is not — because it uses a
> smoother (`relax_type`) that does not stay symmetric/positive-definite when run in parallel.**

Why the smoother controls this: the V-cycle is SPD only if its smoother is symmetric and
positive-definite. The default smoother is **hybrid Gauss–Seidel** (`relax_type=8`): inside one
rank/thread it is real Gauss–Seidel, but couplings that cross a rank/thread boundary are handled
Jacobi-style with stale values. That makes the smoother — and therefore $M$ — **lose
positive-definiteness once the work is split into ≳3 pieces** (ranks × threads). CG detects this as
`gamma = (r, M⁻¹r) ≤ 0` (a quantity that *must* be positive for an SPD $M$) and bails. It is exactly
the "negative pivot / AMG operator not SPD" found by eigendecomposition — same event, two names.

**So it is not a bug in `A`, the partition quality, or Hypre's MPI code. It is a smoother choice:
`relax_type=8` is not SPD-robust under parallelism.** The fix is to pick a smoother that *is*.

Which smoothers are SPD-safe? We tested **all 22** implemented `relax_type` values (issue notebook,
EXP-7). Only **two** both converge and stay SPD under parallel fragmentation:

| relax_type | smoother | SPD-safe in parallel? |
|--:|---|:--:|
| 16 | **Chebyshev** | ✅ (also the fastest) |
| 18 | **l1-Jacobi** | ✅ |
| 8 (default), 6, 3, 4, 0, 7, 17, … | hybrid GS family, plain Jacobi, FCF | ❌ goes non-SPD → bail |
| 1, 2, 13, 14, 15 | one-directional / sequential GS, CG-smoother | ❌ not symmetric → never converges |

Chebyshev and l1-Jacobi are SPD-safe because neither has the "hybrid" cross-boundary asymmetry:
Chebyshev is a polynomial in `A` (only matvecs — identical regardless of partition), l1-Jacobi is
purely diagonal (no neighbor coupling at all).
""")

# ===================================================================== Sec 2 relax sweep
md(r"""
## 2. Fixing the baseline: `relax_type` sweep

Baseline = the failing config from `analysis_2026-5-21`: **`row_block + pcg + boomeramg`, dim=1, no
nodal partition.** We change *only* the smoother and measure iterations + time at np=1 and np=8 on
the hard matrix `mat_twist/3146/71_1` (`tol=1e-10`, `OMP_NUM_THREADS=1`).

`time` = `elapse_time` (analyze + factorize + solve, rank 0), the same metric as the 5-21 "wall".
""")
code("# analysis/subnormal_fix_configs.sh  (SECTION 2)  -- mat_twist/3146/71_1, OMP=1",
out=r"""  np  relax_type            | iters | time(s) | verdict
  --  --------------------- | ----- | ------- | -------
   1  8  (default hybrid-GS)|  544  |  1.11   | ok
   1  16 (Chebyshev)        |  886  |  1.56   | ok
   1  18 (l1-Jacobi)        | 1324  |  1.30   | ok
   8  8  (default hybrid-GS)|    5  |  0.048  | BAIL  <- non-SPD
   8  16 (Chebyshev)        |  909  |  0.985  | ok
   8  18 (l1-Jacobi)        | 1323  |  0.754  | ok
""")
md(r"""
**Reading Section 2:**

| relax_type | np=1 | np=8 | verdict |
|--:|---|---|---|
| **8** (default, hybrid-GS) | 544 it / 1.11 s ✅ | **BAIL** ❌ | unusable at np≥3 |
| **16** (Chebyshev) | 886 it / 1.56 s ✅ | 909 it / 0.985 s ✅ | works |
| **18** (l1-Jacobi) | 1324 it / 1.30 s ✅ | 1323 it / 0.754 s ✅ | works, **fastest at np=8** |

- The default `relax_type=8` **bails at np=8** — the bug. Both 16 and 18 fix it.
- **Iterations vs wall time disagree:** Chebyshev needs *fewer iterations* (909 vs 1323) but is
  *slower in wall time* (0.985 vs 0.754 s). Each Chebyshev step costs ~2 matvecs (degree-2 polynomial)
  plus a per-level eigenvalue-estimation setup, so its lower iteration count doesn't pay off here.
- **Iteration count is essentially np-independent** for both (e.g. 18: 1324→1323), confirming they are
  partition-robust SPD preconditioners.

**Recommendation for the baseline (among 8 / 16 / 18): `relax_type=18` (l1-Jacobi)** — it both fixes
the SPD bug and is the fastest by wall time. (Chebyshev is the choice if you optimize iteration
count / memory rather than wall time.) Section 3 shows the better dim=3 stack.
""")

# ===================================================================== Sec 3 best config
md(r"""
## 3. Best Hypre configuration for our benchmark

Our benchmark projects every system to PSD, so **CG is always valid — we prefer PCG over GMRES**
wherever it works. This section searches for the best PCG stack and compares it to the best GMRES
stack from 5-21.

All runs: `mat_twist/3146/71_1`, **dim=3 + node-aligned partition** (nodal coarsening, the right AMG
for 3-DOF elasticity — 5-21 §3.1), np=8, `OMP_NUM_THREADS=1`, `tol=1e-10`.

### 3a. PCG config search

Axes: partition ∈ {row_block, metis} × preconditioner ∈ {BoomerAMG (relax 16 / 18), Euclid ILU}.
""")
code("# analysis/subnormal_fix_configs.sh  (SECTION 3a)  -- pcg, dim=3, node-aligned, np=8",
out=r"""  partition  precond     smoother      | iters | time(s) | verdict
  ---------  ----------  ------------  | ----- | ------- | -------
  row_block  boomeramg   Chebyshev(16) |  438  |  0.620  | ok
  row_block  boomeramg   l1-Jacobi(18) |  563  |  0.469  | ok
  row_block  euclid(ILU) --            | 2098  |  0.358  | ok
  metis      boomeramg   Chebyshev(16) |  440  |  0.407  | ok
  metis      boomeramg   l1-Jacobi(18) |  562  |  0.279  | ok   <- best PCG
  metis      euclid(ILU) --            |    2  |  0.066  | BAIL
""")
md(r"""
**Reading 3a:**

- **Best PCG = `metis + boomeramg + l1-Jacobi(18)`, dim=3 nodal: 562 iter / 0.279 s.** metis (vs
  row_block) helps every BoomerAMG config (0.469→0.279 for l1-Jacobi) by keeping coarsening
  connectivity-aware; dim=3 nodal coarsening roughly halves iterations vs dim=1 (Sec 2).
- **l1-Jacobi(18) beats Chebyshev(16) on wall time** again (0.279 vs 0.407 s with metis), same reason
  as Sec 2 — fewer Chebyshev iterations don't offset its higher per-iteration + setup cost.
- **Euclid is partition-sensitive:** `row_block + euclid` works (0.358 s) but `metis + euclid`
  **bails** — metis makes each rank's local block more coherent, which destabilizes block-Jacobi ILU
  (matches 5-21 §3). So Euclid only pairs with row_block.
""")

md(r"""
### 3b. GMRES comparison (and does a better smoother speed it up?)

The 5-21 best GMRES stack was **`metis + gmres + boomeramg`, dim=3 + nodal, relax_type=8**
(§3.1.5: 256 iter / 0.27 s at np=8). First we confirm it, then test whether swapping the smoother to
Chebyshev/l1-Jacobi changes it. (GMRES tolerates a non-SPD `M`, so relax=8 is *legal* here — the
question is purely speed.)
""")
code("# analysis/subnormal_fix_configs.sh  (SECTION 3b)  -- gmres, metis, dim=3, node-aligned, np=8",
out=r"""  smoother        | iters | time(s) | verdict
  --------------  | ----- | ------- | -------
  8  hybrid-GS    |  256  |  0.243  | ok   <- best GMRES (matches 5-21 §3.1.5)
  16 Chebyshev    |  629  |  0.630  | ok
  18 l1-Jacobi    | 1102  |  0.583  | ok
""")
md(r"""
**Reading 3b:**

- **Confirmed:** the 5-21 best GMRES stack `metis + gmres + boomeramg`, dim=3 nodal, **relax_type=8**
  reproduces at **256 iter / 0.243 s** (5-21 reported 256 / 0.27 s ✅).
- **Swapping the smoother to 16/18 makes GMRES *slower*** (256 → 629 / 1102 iter). Reason: hybrid-GS
  (8) is a genuinely *stronger* smoother than Chebyshev/Jacobi, so it needs fewer iterations — and
  GMRES doesn't care that hybrid-GS yields a non-SPD `M` (only PCG does). So for GMRES, **keep
  relax_type=8.** (The SPD-safe smoothers are only *needed* for PCG.)
""")

# ===================================================================== overview table
md(r"""
### 3c. Overview table — all dim=3 + node-aligned configs (np=1 and np=8)

Single view of 3a + 3b, `mat_twist/3146/71_1`, `tol=1e-10`, `OMP_NUM_THREADS=1`. `time` =
elapse_time (analyze+factorize+solve). ✅ = converged to 1e-10; ❌ = subnormal/non-SPD bail.

| krylov | partition | precond | smoother | np=1 iter | np=1 time | np=1 | np=8 iter | np=8 time | np=8 |
|---|---|---|---|---:|---:|:--:|---:|---:|:--:|
| PCG | row_block | boomeramg | hybrid-GS(8) — *baseline* | 206 | 0.67 s | ✅ | 3 | 0.074 s | ❌ *(the bug)* |
| PCG | row_block | boomeramg | Chebyshev(16) | 430 | 1.16 s | ✅ | 438 | 0.620 s | ✅ |
| PCG | row_block | boomeramg | l1-Jacobi(18) | 567 | 0.80 s | ✅ | 563 | 0.469 s | ✅ |
| PCG | row_block | euclid ILU | — | 1 | 0.15 s | ❌ | 2098 | 0.358 s | ✅ |
| PCG | **metis** | boomeramg | Chebyshev(16) | 430 | 1.18 s | ✅ | 440 | 0.407 s | ✅ |
| **PCG** | **metis** | **boomeramg** | **l1-Jacobi(18)** | 567 | 0.80 s | ✅ | **562** | **0.279 s** | ✅ |
| PCG | metis | euclid ILU | — | 1 | 0.16 s | ❌ | 2 | 0.066 s | ❌ |
| **GMRES** | **metis** | **boomeramg** | **hybrid-GS(8)** ⭐ *formal best* | 273 | 0.97 s | ✅ | **256** | **0.243 s** | ✅ |
| GMRES | metis | boomeramg | Chebyshev(16) | 626 | 1.90 s | ✅ | 629 | 0.630 s | ✅ |
| GMRES | metis | boomeramg | l1-Jacobi(18) | 1151 | 1.81 s | ✅ | 1102 | 0.583 s | ✅ |

⭐ **formal best** = `metis + gmres + boomeramg + hybrid-GS(8)` (0.243 s) — the fastest config overall.
**Bold PCG row** = best *PCG* stack (`metis + pcg + boomeramg + l1-Jacobi`, 0.279 s), preferred for our
PSD-projected benchmark since it keeps the SPD-correct CG method (only ~15 % slower than the formal best).
The top row is the original **baseline** (`row_block + pcg + boomeramg + hybrid-GS`) — fine at np=1 but
**bails at np=8**: that is exactly the bug this notebook fixes.

Quick read of the table:
- **The baseline bails at np=8** (3 iters, non-SPD) even with dim=3 nodal coarsening — nodal coarsening
  makes the preconditioner *stronger* but does not fix the smoother's SPD loss; only the smoother swap does.
- **At np=1 `row_block` and `metis` are identical** (430/430, 567/567) — with one rank there is no
  partitioning, so the partition axis only matters at np≥2. metis's payoff shows at np=8 (e.g.
  l1-Jacobi 0.469→0.279 s).
- **Euclid only works with `row_block`** and only at np≥2 (it bails at np=1, and `metis` scrambles its
  block-Jacobi ILU → bail). Useful but fragile.
- **GMRES wants hybrid-GS(8)** (256 it); giving GMRES the SPD-safe smoothers just slows it down
  (629/1102 it). PCG wants l1-Jacobi/Chebyshev (it *must* — hybrid-GS bails).
- **l1-Jacobi(18) < Chebyshev(16) in wall time** everywhere (e.g. metis np=8: 0.279 vs 0.407 s),
  though Chebyshev always takes fewer iterations.
""")

# ===================================================================== recommendation
md(r"""
## Recommendation

Putting the best of each stack side by side (`mat_twist/3146/71_1`, np=8, dim=3 nodal, OMP=1):

| stack | iters | time(s) | SPD-clean PCG? |
|---|---:|---:|:--:|
| **`metis + pcg + boomeramg + l1-Jacobi(18)`** | 562 | **0.279** | ✅ **yes** |
| `metis + pcg + boomeramg + Chebyshev(16)` | 440 | 0.407 | ✅ yes |
| `row_block + pcg + euclid(ILU)` | 2098 | 0.358 | ✅ yes (ILU) |
| `metis + gmres + boomeramg + hybrid-GS(8)` | 256 | 0.243 | ❌ GMRES |

**For our PCG-based benchmark (we project to PSD, so CG is always valid and preferred):**

> **Use `metis + pcg + boomeramg`, `dim=3` + node-aligned, `relax_type=18` (l1-Jacobi).**
> 562 iter / 0.279 s at np=8 — the fastest *PCG* config, SPD-clean, and only ~15 % slower than the
> best GMRES stack while keeping the simpler, textbook-correct CG method.

Notes:
- **Smoother:** `relax_type=18` (l1-Jacobi) is the wall-time winner for PCG; `16` (Chebyshev) is the
  iteration-count winner if memory/iters matter more than wall time. **Never the default `8`** with
  PCG at np≥3 (it bails). For **GMRES**, keep `8` (fastest there).
- **Partition:** `metis` for BoomerAMG; `row_block` if you use Euclid (metis breaks Euclid's BJ-ILU).
- **dim=3 + node-aligned** is worth it: ~2× fewer iterations than dim=1 for this 3-DOF elasticity
  problem.
- GMRES(relax=8) remains marginally fastest overall; keep it as the non-PCG fallback / for non-SPD
  (adjoint, mixed) systems.

*(All env-driven via `POLYSOLVE_HYPRE_{PARTITION,PRECOND,KRYLOV,RELAX,DIM,NODE_ALIGNED}` on the
`fix_subnormal_hypre` branch; `RELAX`/`MAXLEVELS` toggles added for this study. Script:
`analysis/subnormal_fix_configs.sh`.)*
""")

nb["cells"] = cells
p = "/u/1/chenyang/benchmark/analysis/analysis_2026-6-2-hypre-subnormal-fix.ipynb"
nbf.write(nb, p)
print("wrote", p, "with", len(cells), "cells")
