#!/usr/bin/env python
# Matrix-level probes for the Hypre subnormal/non-SPD investigation (2026-06-02).
# Run with: ~/miniconda3/envs/benchmark/bin/python analysis/subnormal_matrix_probe.py
import sys
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh, splu

DATADIR = "/mnt/hdd1/chenyang/benchmark_data/larger_matrix_exp/mat_twist/trial_1_result_part1/3146"


def load_bin(path):
    """Reader for PolyFEM SerializeStiffnessMatrix (LARGE_INDEX/int64) format."""
    buf = open(path, "rb").read()
    off = 0

    def take(dt, n=1):
        nonlocal off
        a = np.frombuffer(buf, dtype=dt, count=n, offset=off)
        off += a.nbytes
        return a

    dim, spd, seq = take(np.int32, 3)
    assert np.frombuffer(buf, np.int32, 1, off)[0] == -1, "expected int64 LARGE_INDEX format"
    off += 4
    rows, cols, nnz, innS, outS = [int(x) for x in take(np.int64, 5)]
    vals = take(np.float64, nnz)
    outer = take(np.int64, outS)
    inner = take(np.int64, nnz)
    indptr = np.empty(cols + 1, np.int64)
    indptr[:cols] = outer
    indptr[cols] = nnz
    A = csc_matrix((vals, inner, indptr), shape=(rows, cols)).tocsr()
    return A, int(dim), int(spd)


def smallest_largest_eig(A):
    """(lambda_min, lambda_max) via shift-invert for the smallest, Lanczos for the largest."""
    lam_max = float(eigsh(A, k=1, which="LA", return_eigenvectors=False, maxiter=5000)[0])
    try:
        lam_min = float(eigsh(A, k=1, sigma=0.0, which="LM", return_eigenvectors=False,
                              maxiter=5000)[0])
    except Exception:
        # near-singular: tiny negative shift
        lam_min = float(eigsh(A, k=1, sigma=-1e-8 * abs(lam_max), which="LM",
                              return_eigenvectors=False, maxiter=5000)[0])
    return lam_min, lam_max


def probe_global(tag, A, spd_flag):
    d = A.diagonal()
    av = np.abs(A.data)
    nz = av[av > 0]
    # subnormal doubles: 0 < |x| < 2.2250738585072014e-308
    TINY = np.finfo(np.float64).tiny  # smallest NORMAL double
    n_subnormal = int(((nz > 0) & (nz < TINY)).sum())
    lam_min, lam_max = smallest_largest_eig(A)
    cond = lam_max / lam_min if lam_min > 0 else float("inf")
    print(f"[{tag}] N={A.shape[0]} nnz={A.nnz} header_spd={spd_flag}")
    print(f"    lambda_min = {lam_min:.6e}   lambda_max = {lam_max:.6e}   cond = {cond:.3e}   SPD={lam_min>0}")
    print(f"    diag: min={d.min():.3e} max={d.max():.3e}  range(max/min)={d.max()/d.min():.3e}  any<=0={ (d<=0).any() }")
    print(f"    |a_ij|: min_nz={nz.min():.3e} max={nz.max():.3e}  subnormal_entries={n_subnormal}  (<1e-30: {int((nz<1e-30).sum())})")
    return dict(lam_min=lam_min, lam_max=lam_max, cond=cond, dmin=float(d.min()),
                dmax=float(d.max()), n_subnormal=n_subnormal)


def probe_rowblock_local_spd(A, nranks):
    """Row-block partition (what HypreSolver RowBlock does): is each rank's
    contiguous diagonal sub-block A[lo:hi, lo:hi] itself SPD?  If a rank owns a
    locally-indefinite block, the local relaxation/smoother on that rank is
    ill-posed -> a mechanism for the parallel V-cycle to lose SPD-ness."""
    N = A.shape[0]
    base, rem = divmod(N, nranks)
    print(f"  -- row-block local diagonal blocks, np={nranks} --")
    worst = None
    for r in range(nranks):
        lo = r * base + min(r, rem)
        hi = lo + base + (1 if r < rem else 0)
        Bd = A[lo:hi, lo:hi]
        lo_eig = float(eigsh(Bd, k=1, sigma=0.0, which="LM",
                             return_eigenvectors=False, maxiter=5000)[0])
        flag = "SPD" if lo_eig > 0 else "INDEFINITE!"
        print(f"     rank {r}: rows[{lo},{hi}) size={hi-lo}  lambda_min={lo_eig:.4e}  {flag}")
        if worst is None or lo_eig < worst:
            worst = lo_eig
    print(f"     worst local lambda_min across ranks = {worst:.4e}")
    return worst


if __name__ == "__main__":
    # Part A: canonical hard matrix deep probe
    print("=" * 70)
    print("PART A — canonical hard matrix 71_1")
    print("=" * 70)
    A, dim, spd = load_bin(f"{DATADIR}/71_1_A.bin")
    probe_global("71_1", A, spd)
    print()
    for npr in (2, 3, 4, 8):
        probe_rowblock_local_spd(A, npr)
        print()

    # Part B: conditioning trend across simulation steps (first newton iter)
    print("=" * 70)
    print("PART B — conditioning trend across steps (step_1)")
    print("=" * 70)
    steps = [1, 5, 10, 15, 20, 25, 28, 30, 31, 32, 35, 40, 50, 60, 71]
    print(f"{'step':>5} {'lam_min':>12} {'lam_max':>12} {'cond':>12} {'dmin':>11} {'dmax':>11}")
    for s in steps:
        import os
        p = f"{DATADIR}/{s}_1_A.bin"
        if not os.path.exists(p):
            print(f"{s:>5}  (missing)")
            continue
        As, _, _ = load_bin(p)
        lm, lM = smallest_largest_eig(As)
        d = As.diagonal()
        print(f"{s:>5} {lm:>12.4e} {lM:>12.4e} {lM/lm:>12.3e} {d.min():>11.3e} {d.max():>11.3e}")
