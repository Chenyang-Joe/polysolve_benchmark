#!/bin/bash
# Serial Hypre experiments for the 2026-06-02 non-SPD/subnormal investigation.
# All on mat_twist/3146. Default config = row_block + pcg + boomeramg, dim=1.
set -u
BIN=/u/1/chenyang/benchmark/build.trilinos_tpetra/TestMatLogger
D=/mnt/hdd1/chenyang/benchmark_data/larger_matrix_exp/mat_twist/trial_1_result_part1/3146
# 30s is enough to detect the subnormal bail (it fires within ~4 iters / <2s);
# a run that survives 30s without bailing has a positive-definite M (gamma>0).
TO=30

run() { # np step  [extra env already exported]
  local np=$1 s=$2
  local out
  out=$(timeout $TO mpirun --oversubscribe -np "$np" "$BIN" "$D/${s}_1_A.bin" "$D/${s}_1_b.bin" Hypre 2>&1)
  local sub it fr
  sub=$(echo "$out" | grep -c "Subnormal gamma")
  it=$(echo "$out"  | grep "num_iterations" | grep -oE "[0-9]+$" | tail -1)
  fr=$(echo "$out"  | grep "final_res_norm"  | grep -oE "[0-9.eE+-]+$" | tail -1)
  [ -z "$it" ] && it="TIMEOUT"
  printf "    np=%-2s step=%-3s iters=%-6s final_res=%-12s subnormal=%s\n" \
         "$np" "$s" "$it" "${fr:-NA}" "$([ "$sub" -gt 0 ] && echo "YES($sub)" || echo no)"
}

echo "############################################################"
echo "# EXP-1  onset vs step (np=4, default smoother relax=8)"
echo "############################################################"
unset POLYSOLVE_HYPRE_RELAX POLYSOLVE_HYPRE_MAXLEVELS
for s in 1 20 30 40 45 50 60 71; do run 4 "$s"; done

echo
echo "############################################################"
echo "# EXP-2  smoother sweep on the hard matrix (step 71, np=4)"
echo "#   relax: 8=l1-sym-GS(default) 6=sym-GS 3=hybrid-GS"
echo "#          18=l1-Jacobi(SPD,partition-indep) 0=Jacobi"
echo "############################################################"
unset POLYSOLVE_HYPRE_MAXLEVELS
for rt in 8 6 3 18 0; do
  export POLYSOLVE_HYPRE_RELAX=$rt
  echo "  relax_type=$rt :"
  run 1 71
  run 4 71
done
unset POLYSOLVE_HYPRE_RELAX

echo
echo "############################################################"
echo "# EXP-3  isolate smoother from hierarchy: max_levels=1"
echo "#   (BoomerAMG becomes a pure smoother, no coarsening/RAP)"
echo "############################################################"
export POLYSOLVE_HYPRE_MAXLEVELS=1
for rt in 8 18; do
  export POLYSOLVE_HYPRE_RELAX=$rt
  echo "  max_levels=1 relax_type=$rt :"
  run 4 71
done
unset POLYSOLVE_HYPRE_RELAX POLYSOLVE_HYPRE_MAXLEVELS

echo
echo "############################################################"
echo "# EXP-4  GMRES vs PCG with the SAME (non-SPD) BoomerAMG"
echo "#   GMRES tolerates non-SPD M; PCG does not. Signature test."
echo "############################################################"
for kry in pcg gmres; do
  export POLYSOLVE_HYPRE_KRYLOV=$kry
  echo "  krylov=$kry (relax=8 default, np=4, step 71):"
  run 4 71
done
unset POLYSOLVE_HYPRE_KRYLOV
echo "ALL DONE"
