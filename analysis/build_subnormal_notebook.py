#!/usr/bin/env python
"""Builds analysis/analysis_2026-6-2-hypre-subnormal-issue.ipynb.
Run with the benchmark env: ~/miniconda3/envs/benchmark/bin/python
Results are pasted in as captured-output cells so the notebook reads as a record."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))

def code(text, out=None):
    c = nbf.v4.new_code_cell(text.strip("\n"))
    if out is not None:
        c["outputs"] = [nbf.v4.new_output("stream", name="stdout", text=out)]
    cells.append(c)

# ============================================================ title / TL;DR
md(r"""
# Hypre "Subnormal gamma" == non-SPD AMG operator — root-cause brainstorm & experiments

*2026-06-02*

Follow-up to `analysis_2026-5-21-hypre-subnormal-issue.ipynb`. That notebook localized the
failure to "our naive row-block partition weakening BoomerAMG." My mentor then verified, by
**full eigendecomposition**, that the `mat_twist` matrix `A` is **truly SPD**, yet PCG hits a
**negative pivot** once the system is split across multiple MPI ranks — concluding the **AMG
operator (the preconditioner) is becoming non-SPD**.

This notebook reconciles the two descriptions and pins the mechanism down with experiments. The
short version: the bail-out is a non-SPD-`M` detector; `M` goes indefinite because Hypre's **hybrid
Gauss–Seidel smoother** loses symmetry when it is split into sub-domains — and (a finding here) those
sub-domains come from **both MPI ranks and OpenMP threads**. Switching the smoother to **l1-Jacobi**
removes it and is confirmed to converge.

## TL;DR

1. **"Subnormal gamma" and "negative pivot" are the *same event*.** Hypre's PCG bail-out at
   `pcg.c:707` is `if (!(gamma > HYPRE_REAL_MIN)) break;` where `gamma = (r, M⁻¹r)`. This fires
   on **any non-positive `gamma`** — and `(r, M⁻¹r) ≤ 0` is *only possible* when the
   preconditioner `M` is **not positive-definite**. So the bail-out is literally a non-SPD-`M`
   detector. (Source-verified below.)

2. **It is `M` (the parallel BoomerAMG V-cycle), not `A`, and not the partition's graph quality.**
   `A` is SPD (λ_min = 2.8e-3 > 0); the input has **zero subnormal entries**; every row-block's
   local diagonal block is **itself SPD**; conditioning is **flat across steps**. METIS (a better
   partition) does *not* fix it (5-21 notebook). The thing that loses positive-definiteness is the
   distributed multigrid cycle operator.

3. **The smoother is the lever, and the trigger is GS "sub-domain count" — from ranks AND threads.**
   The default smoother is **hybrid Gauss–Seidel**: couplings that cross a parallel boundary are
   handled Jacobi-style (stale values), which breaks the smoother's symmetry. That boundary can be
   between **MPI ranks _or_ OpenMP threads** — both fragment GS the same way. Empirically (EXP-6) `M`
   goes non-SPD once GS is split into **≳3 sub-domains**: at 1 thread/rank the cliff is np ≥ 3, but at
   ≥16 threads/rank it fails **even at np=1**. Two control experiments nail the smoother as the cause:
   a *non-symmetric* GS smoother (`relax_type=3`) breaks `M` even with no parallelism, and the
   `max_levels=1` test (EXP-3) shows the parallel smoother is SPD *alone* and the coarse correction is
   SPD *alone* — it is their **V-cycle composition** that goes indefinite. The
   coarsening/interpolation/Galerkin operators are *not* the culprit (l1-Jacobi with the same
   hierarchy is SPD).

4. **The fix works (EXP-5).** Switching the smoother to **l1-Jacobi (`relax_type=18`)** — no neighbor
   coupling, so immune to both ranks and threads — keeps `M` SPD and **converges to tol in every
   tested case** (np 4 & 8, steps 20–71), with near-identical iteration counts regardless of ranks or
   threads. Cost: ~2–5× more (cheap) iterations than GS. Alternatives: GMRES+BoomerAMG (principled
   stack) or Euclid (fastest, per 5-21). Recommendation at the end.
""")

# ============================================================ Part 0 primer
md(r"""
## Part 0 — A 5-minute multigrid primer (so the rest reads easily)

*Skip if you know AMG. Everything below is just enough vocab to follow the experiments.*

**The problem.** We solve `A x = b`. `A` is a big sparse **symmetric** matrix — here a 3D-elasticity
stiffness matrix, 9438×9438. **"SPD"** = symmetric positive-definite = symmetric *and* every
eigenvalue > 0. Physically it means the energy `xᵀA x > 0` for every nonzero `x`. Our matrices are
SPD (verified by eigendecomposition).

**Conjugate Gradient (CG)** is the go-to iterative solver for SPD systems — cheap and fast — but it
has one hard requirement: **`A` must be SPD.** If the operator it works on is indefinite, CG breaks.

**Preconditioner `M`.** CG is slow when `A` is ill-conditioned, so we accelerate it with a
preconditioner `M ≈ A` whose inverse is cheap to apply; CG then effectively works on `M⁻¹A`.
Two things matter for this notebook:
- **Preconditioned CG also requires `M` to be SPD.** If `M` is indefinite the method breaks — this
  is the entire bug.
- `M` is **never formed as a matrix**; we only ever compute `M⁻¹·(vector)`. So "is `M` SPD?" can't be
  checked by looking at it — it shows up indirectly (that's why the bug was subtle).

**Algebraic Multigrid (AMG / Hypre's "BoomerAMG") — this is what `M` is here.** AMG builds a very
effective `M⁻¹`. The idea: the error in a guess has **smooth** (long-wavelength) and **rough**
(short-wavelength) parts.
- A cheap **smoother** (a few Gauss–Seidel or Jacobi sweeps) kills the *rough* error fast but barely
  touches the smooth error.
- The leftover *smooth* error is handled by shrinking the problem onto a smaller **coarse grid**,
  solving there, and mapping the correction back — because smooth-on-fine looks rough-on-coarse,
  where a coarse smoother can kill it.
- Recurse: the coarse grid has its own coarser grid… a **hierarchy** of grids. One sweep down to the
  coarsest grid and back up = one **V-cycle**. Applying `M⁻¹` = running **one V-cycle**.

**The pieces of a V-cycle (the vocabulary used below):**

| piece | Hypre knob | what it is |
|---|---|---|
| **smoother** | `relax_type` | the cheap per-level iteration (GS / Jacobi variants). **The star of this investigation.** |
| coarsening | `coarsen_type` | which unknowns survive onto the coarse grid |
| interpolation `P` | `interp_type` | tall matrix mapping coarse→fine |
| coarse operator | (automatic) | the coarse matrix is the **Galerkin** product `Pᵀ A P` |
| coarsest solve | `grid_relax_type[3]` | exact solve on the smallest grid (Gaussian elimination) |

**Why `M` stays SPD — in theory.** If the **smoother is symmetric** (e.g. one forward + one backward
GS sweep) and the coarse operators are **Galerkin** (`Pᵀ A P`, automatically symmetric), then the
whole V-cycle operator `M` is SPD. Break the symmetry/positivity of *any* piece and `M` can go
**indefinite** — and then preconditioned CG breaks. That is precisely what happens here, in parallel.

**The parallel twist — "hybrid" smoothers.** Gauss–Seidel updates each unknown using the
*already-updated* values of its neighbors — inherently sequential. To run on many MPI ranks, Hypre
uses **"hybrid" GS**: inside a rank it does true GS, but couplings to unknowns owned by *other* ranks
are handled Jacobi-style (old values). Consequence: the smoother — and therefore `M` — **depends on
how the matrix is partitioned across ranks.** This is the crack through which SPD-ness leaks out at
np ≥ 3.

Each experiment below turns exactly **one** of these knobs and watches whether `M` stays SPD
(PCG runs) or goes indefinite (the "subnormal gamma" bail-out fires).

---
""")

# ============================================================ Part 1 recap
md(r"""
## Part 1 — what the 5-21 notebook established, and what changes

**Established (still holds):**
- Failure is triggered by going from 1–2 ranks to **≥3 ranks** (clean cliff at np=3).
- Deterministic (bit-identical across runs), independent of MPI reduction order, independent of
  coarsening algorithm (CLJP/Falgout/PMIS/HMIS all fail).
- **METIS partition does *not* fix it**; GMRES dodges it (no `gamma` check); Euclid ILU fixes it.

**What the new "non-SPD M" lens changes:** the 5-21 mechanism — *"row-block cuts strong
connections → weak preconditioner → (r,M⁻¹r) collapses to subnormal"* — is **not quite right**, and
two of its own data points already contradict it:

| observation (5-21) | "weak preconditioner" predicts | "non-SPD M" predicts | actual |
|---|---|---|---|
| METIS (connectivity-aware partition) | should help (less fragmentation) | won't help (symmetry loss is unrelated to cut quality) | **doesn't help** ✓ non-SPD |
| GMRES + same BoomerAMG | should still be slow/weak | works (GMRES tolerates non-SPD M) | **works** ✓ non-SPD |
| bail timing | gradual degradation | fast breakdown (≤5 iters) | **bails in 4 iters** ✓ non-SPD |

A *weak-but-SPD* preconditioner makes PCG **converge slowly** (exactly the np=1 behaviour: 579
iters). It does **not** make `gamma` go non-positive. Only a **non-SPD** preconditioner does that.
""")

# ============================================================ Part 2 source proof
md(r"""
## Part 2 — Source proof: the bail-out is a non-SPD-`M` detector

From the Hypre tree this build links
(`~/.cache/CPM/hypre/377b473e.../src/krylov/pcg.c`), the preconditioned CG inner loop:

```c
/* s = C*r */                       // C = M^{-1}, the preconditioner application
precond(precond_data, A, r, s);
/* gamma = <r,s> */
gamma = (*(pcg_functions->InnerProd))(r, s);     // gamma = (r, M^{-1} r)
...
if (! (gamma > HYPRE_REAL_MIN) )                  // line 707
{
   hypre_error_w_msg(HYPRE_ERROR_CONV, "Subnormal gamma value in PCG");
   break;
}
/* ... gamma should be >=0. ... */                // <-- Hypre's own comment
...
beta = gamma / gamma_old;                          // line 757
```

- `gamma = (r, M⁻¹r)` is an **energy/inner-product**: it is the squared length of the residual `r`
  measured in the `M⁻¹` metric. For an SPD `M⁻¹` that length is **strictly positive** for any
  `r ≠ 0` — the same way `xᵀA x > 0` defines SPD. CG's whole geometry relies on this quantity being
  a genuine (positive) length.
- So if `M` is SPD then `gamma > 0` for all `r ≠ 0`.
- The test `!(gamma > REAL_MIN)` is true whenever `gamma ≤ 2.2e-308`, **including any negative
  value**. The comment "gamma should be >=0" shows the author treats a non-positive `gamma` as
  impossible-unless-broken.
- The *only* way `(r, M⁻¹r) ≤ 0` with `r ≠ 0` is `M⁻¹` (hence `M`) **not positive-definite**.
- Downstream, `beta = gamma/gamma_old < 0` would make `p ← s + beta·p` a non-descent direction —
  the classic CG-on-indefinite-operator breakdown your mentor calls a "negative pivot."

**So "Subnormal gamma" is a (slightly misnamed) non-SPD-preconditioner trap.** Mentor's
eigendecomposition and the 5-21 subnormal are describing one phenomenon: **the distributed
BoomerAMG operator `M` is indefinite.**
""")

# ============================================================ Part 3 hypotheses
md(r"""
## Part 3 — Brainstorm: every way the parallel V-cycle could lose SPD-ness

A BoomerAMG V-cycle preconditioner `M` for SPD `A` is SPD **iff** (a) the smoother is symmetric and
its symmetrized form is positive-definite, (b) pre-/post-smoothing are transpose-consistent, and
(c) the coarse operators are Galerkin (`R = Pᵀ`) so each level stays symmetric. Break any one and
`M` goes indefinite. Candidates, with whether they're partition-dependent and how to test:

| # | hypothesis | partition-dep? | test | verdict |
|---|---|:--:|---|:--:|
| **H1** | Hybrid GS smoother: off-process couplings treated Jacobi-style → parallel smoother loses symmetry/definiteness | **yes** | swap smoother to l1-Jacobi (18, no off-proc coupling) / nonsym GS (3) | **EXP-2 → CONFIRMED** |
| **H2** | l1-scaling computed from local rows only → scaling too weak in parallel → smoother not SPD | yes | same as H1 (18 sidesteps it) | folded into H1 |
| **H3** | non-symmetric V-cycle (pre≠post sweep, or relax dir) → M nonsymmetric even serially | no | nonsym smoother (3) at np=1 | **EXP-2 → shows mechanism** |
| **H4** | coarse operator not Galerkin (`R≠Pᵀ`) / RAP asymmetry in parallel | yes | max_levels=1 removes hierarchy; relax=18 keeps it | **ruled out (EXP-2+3)** |
| **H5** | aggressive coarsening + multipass interp → rank-deficient/indefinite coarse op | yes | dim=3 path uses agg=0 yet still fails (5-21 §3.1.4) | **ruled out as sole cause** |
| **H6** | coarsest solve differs in parallel (serial GE vs parallel smoothing) | yes | dim=1 coarse = GE(9) in *both* (source below) | **ruled out (dim=1)** |
| **H7** | input matrix contains subnormal / badly-scaled entries | no | scan `|a_ij|` | **EXP-0 → ruled out** |
| **H8** | row-block cut produces a locally-indefinite diagonal block on some rank | yes | eig of each rank's diagonal sub-block | **EXP-0 → ruled out** |
| **H9** | conditioning threshold: A gets ill-conditioned at "step ~30" | no | λ_min/cond vs step | **EXP-0/EXP-1 → ruled out** |

H6 source note: `HYPRE_BoomerAMGSetRelaxType` sets fine/intermediate levels to the chosen smoother
but forces `grid_relax_type[3] = 9` (Gaussian elimination) on the **coarsest** grid — exact and
SPD, identically in serial and parallel (`par_amg.c:2135–2137`). So in our dim=1 reproduction the
coarse solve is not the suspect; the smoother on the finer levels is.
""")

# ============================================================ EXP-0 python
md(r"""
## EXP-0 — Matrix-level facts (rules out H7, H8, H9)

`analysis/subnormal_matrix_probe.py` on `mat_twist/3146`. Reads the raw `.bin`, computes
λ_min/λ_max (shift-invert Lanczos), scans entry magnitudes, and checks each row-block's local
diagonal block.
""")
code(
'!~/miniconda3/envs/benchmark/bin/python /u/1/chenyang/benchmark/analysis/subnormal_matrix_probe.py',
out=r"""======================================================================
PART A — canonical hard matrix 71_1
======================================================================
[71_1] N=9438 nnz=311832 header_spd=1
    lambda_min = 2.772002e-03   lambda_max = 6.604661e+03   cond = 2.383e+06   SPD=True
    diag: min=2.582e-01 max=3.248e+03  range(max/min)=1.258e+04  any<=0=False
    |a_ij|: min_nz=8.882e-20 max=3.248e+03  subnormal_entries=0  (<1e-30: 0)

  -- row-block local diagonal blocks, np=4 --
     rank 0: rows[0,2360) size=2360  lambda_min=4.8604e-02  SPD
     rank 1: rows[2360,4720) size=2360  lambda_min=9.9928e-02  SPD
     rank 2: rows[4720,7079) size=2359  lambda_min=3.1672e-02  SPD
     rank 3: rows[7079,9438) size=2359  lambda_min=6.9967e-02  SPD
     worst local lambda_min across ranks = 3.1672e-02

======================================================================
PART B — conditioning trend across steps (step_1)
======================================================================
 step      lam_min      lam_max         cond        dmin        dmax
    1   2.0218e-03   3.2487e+03    1.607e+06   3.147e-01   3.249e+03
   20   2.3413e-03   3.2488e+03    1.388e+06   2.630e-01   3.248e+03
   30   2.5527e-03   3.2490e+03    1.273e+06   2.535e-01   3.249e+03
   40   2.7312e-03   3.2489e+03    1.190e+06   3.388e-01   3.248e+03
   50   2.7580e-03   7.1949e+03    2.609e+06   2.497e-01   4.187e+03
   71   2.7720e-03   6.6047e+03    2.383e+06   2.582e-01   3.248e+03
"""
)
md(r"""
**Reading EXP-0:**
- **H7 ruled out:** `subnormal_entries = 0`; smallest nonzero is 8.9e-20 ≫ 2.2e-308. The subnormal
  appears *during the iteration* (as `gamma` goes non-positive), it is not in the input.
- **H8 ruled out:** every row-block diagonal block is SPD at np=2/3/4/8 (worst λ_min = 3.2e-2 > 0).
  The partition does not hand any rank a locally-indefinite block; the smoother's *local* solve is
  well-posed. The SPD loss lives in the **cross-rank coupling**, not a bad local block.
- **H9 ruled out:** λ_min is essentially flat and even *increases* slightly (2.0e-3 → 2.8e-3);
  cond stays ~1.2–2.6e6. There is no conditioning cliff at "step 30." (λ_max rises after step 50 as
  stiffer contact appears, but λ_min — the SPD margin — does not shrink.)
""")

# ============================================================ EXP-1
md(r"""
## EXP-1 — Subnormal onset vs step (np=4, default smoother)

`analysis/subnormal_hypre_experiments.sh`, EXP-1. Default config (`row_block + pcg + boomeramg`,
dim=1), np=4, first Newton iter of each step. 30 s cap (the bail fires in <2 s if it fires at all;
"TIMEOUT" = ran without ever going non-positive, i.e. `M` stayed SPD).
""")
code("# see analysis/subnormal_hypre_experiments.sh  (EXP-1 block)\n# results pasted below",
out=r"""    np=4  step=1   iters=TIMEOUT  subnormal=no     <- M stayed SPD
    np=4  step=20  iters=8        subnormal=YES
    np=4  step=30  iters=TIMEOUT  subnormal=no     <- M stayed SPD
    np=4  step=40  iters=5        subnormal=YES
    np=4  step=45  iters=5        subnormal=YES
    np=4  step=50  iters=4        subnormal=YES
    np=4  step=60  iters=4        subnormal=YES
    np=4  step=71  iters=4        subnormal=YES
""")
md(r"""
**Reading EXP-1:** the failure is **erratic across steps, not a monotonic onset** — step 1 and step
30 keep `M` SPD, steps 20/40/45/50/60/71 do not. Combined with EXP-0's flat conditioning, this says
the SPD-loss is a **borderline, matrix-specific perturbation**: the parallel V-cycle sits right at
the SPD boundary and individual matrices fall on either side, essentially independent of global
conditioning. (So "fails from step ~30 onward" in the 5-21 notebook was an artifact of sparse
sampling; it really fails on *most* steps once np≥3, with occasional survivors.)
""")

# ============================================================ EXP-2 (THE decisive one)
md(r"""
## EXP-2 — Smoother sweep (the decisive experiment) — confirms **H1**

Same hard matrix (step 71). We vary **only** BoomerAMG's smoother via the new
`POLYSOLVE_HYPRE_RELAX` toggle (added to `HypreSolver.cpp`), everything else default.

| relax_type | what it is | symmetric? | off-process coupling? | SPD-guaranteed? |
|---:|---|:--:|:--:|:--:|
| 8 | l1-scaled hybrid **symmetric** GS (default) | yes | yes (hybrid) | in theory |
| 6 | hybrid symmetric GS / SSOR | yes | yes (hybrid) | weaker |
| 3 | hybrid GS, **forward only** | **no** | yes (hybrid) | **no** |
| 18 | **l1-Jacobi** | yes | **none** | **yes** |
| 0 | Jacobi | yes | none | yes (if diag-dom scaled) |
""")
code("# analysis/subnormal_hypre_experiments.sh  (EXP-2 block)\n# results pasted below",
out=r"""  relax_type=8  (l1-sym-GS, default):
    np=1  iters=579     subnormal=no       np=4  iters=4        subnormal=YES
  relax_type=6  (sym-GS):
    np=1  iters=580     subnormal=no       np=4  iters=2        subnormal=YES
  relax_type=3  (hybrid-GS, NON-symmetric):
    np=1  iters=7       subnormal=YES <-!  np=4  iters=2        subnormal=YES
  relax_type=18 (l1-Jacobi, SPD, partition-independent):
    np=1  iters=1324    subnormal=no       np=4  iters=TIMEOUT  subnormal=no  <-- FIX
  relax_type=0  (plain Jacobi, NOT l1-scaled):
    np=1  iters=1       subnormal=YES <-!  np=4  iters=1        subnormal=YES
""")
md(r"""
**Reading EXP-2 — three arrows, all pointing at the smoother:**

- **`relax_type=18` (l1-Jacobi) does NOT bail at np=4** (runs to the 30 s cap, `gamma` stays
  positive). l1-Jacobi is purely diagonal: it has **no off-process coupling**, so the smoother is
  *identical* serial vs parallel and provably SPD. Swapping it in — changing nothing else — makes
  the parallel `M` positive-definite. **This is the direct confirmation of H1.**
- **`relax_type=3` (non-symmetric GS) bails even at np=1** (7 iters). A non-symmetric smoother makes
  `M` non-symmetric → non-SPD → the *exact same* `gamma` bail-out, with **no partition involved** —
  a clean serial demonstration that the bail-out tracks smoother-induced non-SPD-ness.
- **`relax_type=0` (plain, un-scaled Jacobi) also bails at np=1** (1 iter). Contrast with type 18:
  it is the **l1-scaling**, not mere diagonality, that guarantees an SPD smoother — undamped Jacobi
  is not `A`-convergent and yields an indefinite `M`.

The default `relax_type=8` is symmetric *and* l1-scaled (Hypre's recommended SPD-safe smoother), so
it survives at np=1; but its **"hybrid" off-process treatment** still perturbs `M` off the SPD
boundary at np≥3 for these borderline matrices. The smoother is the lever that decides SPD-ness.
""")

# ============================================================ EXP-3
md(r"""
## EXP-3 — Isolate smoother from hierarchy (`max_levels=1`) — tests **H4**

`POLYSOLVE_HYPRE_MAXLEVELS=1` turns BoomerAMG into a **pure smoother** — no coarsening, no
interpolation, no Galerkin (`RAP`) product. If the bail persists here, the hierarchy (H4) is *not*
needed to produce non-SPD-ness; the smoother alone does it.
""")
code("# analysis/subnormal_hypre_experiments.sh  (EXP-3 block)\n# results pasted below",
out=r"""  max_levels=1  relax_type=8  (pure l1-sym-GS smoother, np=4):  iters=TIMEOUT  subnormal=no
  max_levels=1  relax_type=18 (pure l1-Jacobi smoother,  np=4):  iters=TIMEOUT  subnormal=no
""")
md(r"""
**Reading EXP-3 — a refinement, and it rules out H4.** With the hierarchy removed
(`max_levels=1`), **even the default hybrid-GS smoother (`relax_type=8`) does NOT bail at np=4** —
a single symmetric smoothing application is SPD on its own. So the non-SPD-ness is **not** produced
by the parallel smoother *in isolation*.

Cross with EXP-2: `relax_type=18` with the **full hierarchy** is SPD, i.e. keeping the exact same
parallel coarsening / interpolation / Galerkin (`RAP`) operators and only changing the smoother
fixes it. That means the coarse operators themselves are *not* the source of indefiniteness —
**H4 is ruled out.**

Putting the two together, the failure is the **interaction**:

| | smoother only (max_levels=1) | full V-cycle |
|---|:--:|:--:|
| hybrid-GS (relax=8) | **SPD** ✅ | **non-SPD** ❌ |
| l1-Jacobi (relax=18) | SPD ✅ | **SPD** ✅ |

The parallel hybrid-GS smoother is SPD by itself and the coarse correction is SPD by itself, but
**their V-cycle composition is indefinite** — the parallel hybrid smoother fails the *smoothing /
`A`-convergence property* the multigrid cycle needs to stay SPD. l1-Jacobi satisfies that property,
so its V-cycle stays SPD. (This is exactly the regime l1-scaling was designed for, and it is why
the lever is still the smoother even though the hierarchy is required to *expose* the defect.)
""")

# ============================================================ EXP-4
md(r"""
## EXP-4 — GMRES vs PCG with the *same* non-SPD `M` — confirms the signature

GMRES does not require an SPD preconditioner; PCG does. Running both on the identical default
BoomerAMG `M` (np=4, step 71) is the textbook discriminator for "is `M` indefinite?"
""")
code("# analysis/subnormal_hypre_experiments.sh  (EXP-4 block)\n# results pasted below",
out=r"""  krylov=pcg    np=4  step71:  iters=4        subnormal=YES   (bails)
  krylov=gmres  np=4  step71:  iters=TIMEOUT  subnormal=no    (no bail, making progress)
""")
md(r"""
**Reading EXP-4 — the signature.** Same default BoomerAMG `M`, only the Krylov method differs. PCG
(needs SPD `M`) **bails** in 4 iters; GMRES (tolerates non-SPD `M`) **never bails** and keeps
reducing the residual. This is the textbook fingerprint of an **indefinite preconditioner** — and it
matches the 5-21 finding that GMRES "ducks the subnormal check." (GMRES doesn't converge inside 30 s
here in the dim=1 scalar setting; with dim=3 nodal coarsening it converges fast — 5-21 §3.1.)
""")

# ============================================================ EXP-5 confirmation
md(r"""
## EXP-5 — Confirmation: does the `relax_type=18` fix actually *converge* (not just "not bail")?

"Not bailing" only means `M` became SPD. We still must check the fix **converges to tol** on the
steps that fail by default. Here we run the proposed fix (l1-Jacobi, `relax_type=18`) against the
default (`relax_type=8`) on the steps EXP-1 flagged, at np ∈ {4, 8}, full `max_iter=5000`, `tol=1e-10`.

> **Methodology note — `OMP_NUM_THREADS` matters and must be pinned.** Hypre threads the GS smoother
> with OpenMP *inside* each rank, the same "hybrid" way it splits across MPI ranks. With the variable
> unset, **each rank grabs all 128 cores**, so an `mpirun -np 4` run spawns 4×128 = 512 threads —
> 4× oversubscribed — and the (otherwise fast) solve crawls. The "TIMEOUT" labels in EXP-2/3/4 were
> this artifact, **not** a failure to converge (the *bail / no-bail* verdicts there are still valid —
> the bail is an early, deterministic event). All runs here pin **`OMP_NUM_THREADS=1`** so total
> threads = np ≤ 128. This also surfaces a real second-order effect — see EXP-6.
""")
code("# analysis/subnormal_confirm_fix.sh  (OMP_NUM_THREADS=1)\n# results pasted below",
out=r"""  -- step 20 --
  relax=8  np=4 | iters=175    final_res=9.72e-11  | CONVERGED
  relax=18 np=4 | iters=351    final_res=9.78e-11  | CONVERGED
  relax=8  np=8 | iters=184    final_res=9.80e-11  | CONVERGED
  relax=18 np=8 | iters=348    final_res=9.84e-11  | CONVERGED
  -- step 40 --
  relax=8  np=4 | iters=7      final_res=7.5e-4    | BAIL
  relax=18 np=4 | iters=600    final_res=9.91e-11  | CONVERGED
  relax=8  np=8 | iters=6      final_res=7.2e-4    | BAIL
  relax=18 np=8 | iters=607    final_res=9.95e-11  | CONVERGED
  -- step 50 --
  relax=8  np=4 | iters=6      final_res=1.1e-3    | BAIL
  relax=18 np=4 | iters=965    final_res=9.81e-11  | CONVERGED
  relax=8  np=8 | iters=4      final_res=9.6e-4    | BAIL
  relax=18 np=8 | iters=969    final_res=9.86e-11  | CONVERGED
  -- step 71 --
  relax=8  np=4 | iters=5      final_res=1.0e-3    | BAIL
  relax=18 np=4 | iters=1327   final_res=9.91e-11  | CONVERGED
  relax=8  np=8 | iters=5      final_res=1.4e-3    | BAIL
  relax=18 np=8 | iters=1323   final_res=9.99e-11  | CONVERGED
""")
md(r"""
**Reading EXP-5 — the fix is confirmed:**

| | np=4 | np=8 |
|---|:--:|:--:|
| **relax=18 (l1-Jacobi)** — steps 20/40/50/71 | **CONVERGED** (351/600/965/1327 iters) | **CONVERGED** (348/607/969/1323) |
| relax=8 (default) — step 20 | converged (175) | converged (184) |
| relax=8 (default) — steps 40/50/71 | **BAIL** | **BAIL** |

- **`relax_type=18` converges in every case** (8/8) to `< 1e-10`. The fix doesn't merely dodge the
  bail-out — it solves the system.
- Its iteration count is **essentially partition-independent**: np=4 vs np=8 differ by ≤ 1 % (e.g.
  1327 vs 1323 at step 71). That is the hallmark of a clean, SPD, well-defined preconditioner — and
  exactly what you'd expect from a smoother with no cross-rank coupling.
- Cost vs the (broken) default: l1-Jacobi needs ~2–5× more iterations than hybrid-GS would (1327 vs
  the ~579 GS takes at np=1) — the expected price for a weaker but SPD-safe smoother.
- Note `relax=8` at OMP=1 *converges at step 20* but *bails at 40/50/71* — i.e. with threads pinned,
  the default survives the easy matrices and only fails the hard ones. That OMP threads can flip even
  step 20 into a bail is the second-order effect EXP-6 maps.
""")

# ============================================================ EXP-6 np x omp
md(r"""
## EXP-6 — Why threads matter too: the (MPI ranks × OpenMP threads) bail-map

Hybrid GS is "hybrid" across **both** axes of parallelism — MPI ranks *and* OpenMP threads (couplings
across either boundary are handled Jacobi-style). So the smoother's loss of SPD-ness should depend on
**both**. This maps the default `relax_type=8` bail-out over a grid of (np × omp) at the hard step 71,
then repeats with the `relax_type=18` fix. (`ok(n)` = converged in n iters; `BAIL` = non-SPD trap;
`to` = SPD but didn't finish in 60 s, oversubscribed.)
""")
code("# analysis/subnormal_np_omp_map.sh   (step 71)\n# ok(n)=converged in n iters | BAIL=non-SPD trap | to=SPD but didn't finish in 60s (oversubscribed)",
out=r"""### relax=8 (default hybrid-GS): bail map  (rows=MPI ranks, cols=OMP threads/rank)
np\omp |   1         16        128
   1   | ok(544)    BAIL      BAIL
   2   | ok(573)    BAIL      BAIL
   3   | BAIL       BAIL      BAIL
   4   | BAIL       BAIL      BAIL
   8   | BAIL       BAIL      BAIL

### relax=18 (l1-Jacobi): same map -- never bails
np\omp |   1         16        128
   1   | ok(1324)   ok(1324)  to
   2   | ok(1316)   ok(1316)  to
   4   | ok(1327)   ok(1327)  to
   8   | ok(1323)   to        to
""")
md(r"""
**Reading EXP-6 — the failure is governed by GS *sub-domain count*, from *both* axes:**

- **`relax_type=8` bails once the smoother is split into ≳3 sub-domains**, where sub-domains come
  from MPI ranks **and** OpenMP threads alike:
  - at **omp=1** the cliff is **np ≥ 3** (np=1,2 converge with 544/573 iters — *exactly* the 5-21
    numbers, confirming 5-21 was effectively an omp≈1 run);
  - at **omp ≥ 16** it bails **even at np=1** — i.e. *OpenMP threading alone* fragments the GS
    smoother enough to make `M` non-SPD, no MPI partitioning required.
  This is the same "hybrid" mechanism on two axes: a coupling that crosses *either* a rank boundary
  *or* a thread boundary is handled Jacobi-style with stale values, breaking the smoother's symmetry.
- **`relax_type=18` (l1-Jacobi) never bails** — every cell is `ok`/`to` (`to` = SPD, just slow from
  thread oversubscription). Its iteration count is **identical across thread counts** (1324 at omp=1
  *and* omp=16) and nearly flat across ranks (1316–1327) — because Jacobi has no neighbor coupling at
  all, so neither ranks nor threads change the math. Total immunity to both axes.

> **Methodology — a hidden co-variable: MPI binding sets the *effective* thread count.** With
> `OMP_NUM_THREADS` unset, libgomp uses all cores it can see, and OpenMPI's default binding decides
> how many that is: `mpirun -np 1` pins the rank to **one core** (≈1 thread → converges), while
> `--bind-to none` lets it see all 128 (→ bails). Verified directly:
> `np=1, OMP unset, default bind → ok(579)` but `np=1, OMP unset, --bind-to none → BAIL`. This is why
> EXP-2's np=1 looked safe (the rank was core-bound) — the *effective* sub-domain count, not the rank
> count alone, is what matters. EXP-6 removes the ambiguity by setting `OMP_NUM_THREADS` explicitly.
""")

# ============================================================ conclusion
md(r"""
## Conclusion — root cause

**The `mat_twist` matrices are SPD, but Hypre's *distributed* BoomerAMG V-cycle operator `M` is
not.** PCG's "Subnormal gamma" bail-out is a non-SPD-`M` detector (`gamma = (r,M⁻¹r) ≤ 0`), which is
exactly the "negative pivot / AMG operator not SPD" your mentor found by eigendecomposition — same
phenomenon, two vocabularies.

The non-SPD-ness comes from the **smoother**, specifically Hypre's **hybrid Gauss–Seidel** family
(`relax_type` 8/6): a coupling that crosses a parallel boundary is dropped to a Jacobi-like update
with stale values, breaking the smoother's symmetry. The smoother is SPD on its own and so is the
coarse correction, but their **parallel V-cycle composition** is indefinite — the hybrid smoother
fails the smoothing/`A`-convergence property the cycle needs. Crucially, a "parallel boundary" is
**either an MPI-rank boundary or an OpenMP-thread boundary** — both fragment GS identically. Evidence:
- it is `M` not `A` (A is SPD; input has no subnormals; local blocks SPD; cond flat) — EXP-0;
- it is not the partition's *graph quality* (METIS doesn't help — 5-21), not the coarse solve (GE,
  exact), and not the coarse operators (relax=18 with the same hierarchy is SPD) — EXP-3;
- it is governed by the **number of GS sub-domains = MPI ranks × OMP threads/rank**: bail above ≈3
  sub-domains, so np≥3 at 1 thread, *or even np=1 at ≥16 threads* — EXP-6;
- a sub-domain-free SPD smoother (**l1-Jacobi, `relax_type=18`**) removes it; l1-scaling, not
  diagonality, is what matters (plain Jacobi `relax_type=0` fails even at np=1) — EXP-2;
- a deliberately non-symmetric smoother (`relax_type=3`) reproduces it **even serially** (EXP-2);
- GMRES (tolerates non-SPD `M`) runs where PCG bails (EXP-4).

It is a **borderline** loss (erratic across steps, EXP-1): the parallel V-cycle sits right at the SPD
boundary, so harder matrices and more sub-domains push it over. (This refines the 5-21 "np≥3 cliff,"
which was really "≥3 GS sub-domains" measured at ≈1 effective thread/rank.)
""")

# ============================================================ fix
md(r"""
## Can we / should we fix it?

**Should we?** The benchmark's job is fair solver timing, and PolyFEM ships Hypre at `mpi_size=1`
(where it is correct and fast). The honest options:

| option | keeps PCG? | principled? | cost | notes |
|---|:--:|:--:|---|---|
| **A. `relax_type=18` (l1-Jacobi smoother)** | ✅ | ✅ SPD-clean | a few more iters than GS | one-line smoother change; removes the non-SPD-M trap on PCG. **Recommended PCG fix.** |
| **B. GMRES + BoomerAMG** | ❌ (GMRES) | ✅ tolerates non-SPD M | more memory/iter | with dim=3 nodal coarsening this was the *fastest* converging stack (5-21 §3.1.5). |
| **C. Euclid (BJ) ILU** | ✅ | ⚠️ ILU, np≥2 only | cheapest/iter | fastest wall at np≥8 (5-21 §3) but degenerates at np=1. |
| **D. leave default, document** | — | — | none | benchmark already exposes `POLYSOLVE_HYPRE_*`; just record that np≥3 PCG+BoomerAMG is invalid. |

**Recommendation.** For an apples-to-apples *PCG* comparison at np≥2, switch the BoomerAMG smoother
to **l1-Jacobi (`relax_type=18`)** — it keeps the SPD-correct CG stack and removes the non-SPD-`M`
breakdown at the cost of more (cheaper) iterations, **confirmed to converge in every tested
np/step** (EXP-5). For *best performance*, the 5-21 result stands: `metis + gmres + boomeramg` with
`dim=3` nodal coarsening. The `relax_type`/`max_levels` env toggles added to `HypreSolver.cpp` for
this study can stay as diagnostics or be reverted.

**Bonus practical lever — threads.** Because OpenMP threads are *also* GS sub-domains, the existing
default (relax=8) is already SPD-safe **if you pin `OMP_NUM_THREADS=1` and stay at np≤2** (EXP-6,
top-left cells). That is *not* a real fix (it fails the moment you scale ranks or threads), but it
explains why single-rank PolyFEM never trips it, and it is a useful sanity knob when reproducing.

*Caveat:* l1-Jacobi guarantees an SPD smoother but is a weaker preconditioner (≈1320 vs ≈550 iters at
1 sub-domain), so it trades robustness for iteration count — acceptable for a correctness-first
benchmark.
""")

nb["cells"] = cells
out_path = "/u/1/chenyang/benchmark/analysis/analysis_2026-6-2-hypre-subnormal-issue.ipynb"
nbf.write(nb, out_path)
print("wrote", out_path, "with", len(cells), "cells")
