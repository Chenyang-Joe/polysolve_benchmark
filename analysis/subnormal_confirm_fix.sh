#!/bin/bash
# Confirmation experiment (2026-06-02): does relax_type=18 (l1-Jacobi smoother)
# not only avoid the non-SPD bail-out but actually CONVERGE to tol on the steps
# that fail with the default smoother?  Compare default relax=8 (bails) vs
# relax=18 (proposed fix) on steps that EXP-1 showed failing, at np=4 and np=8.
# Full max_iter (5000, from linear-solver-spec.json); generous timeout.
set -u
BIN=/u/1/chenyang/benchmark/build.trilinos_tpetra/TestMatLogger
D=/mnt/hdd1/chenyang/benchmark_data/larger_matrix_exp/mat_twist/trial_1_result_part1/3146
TO=90
# CRITICAL: pin 1 OpenMP thread per rank. Unset, each rank grabs all 128 cores,
# so np=4 spawns 4x128 threads (4x oversubscription) and the solve crawls -- an
# artifact that has nothing to do with the solver. With OMP=1, np*1 <= 128 cores.
export OMP_NUM_THREADS=1

run() { # relax np step
  local rt=$1 np=$2 s=$3
  local t0 t1 out sub it fr conv
  t0=$(date +%s.%N)
  out=$(POLYSOLVE_HYPRE_RELAX=$rt timeout $TO mpirun --oversubscribe -np "$np" \
        "$BIN" "$D/${s}_1_A.bin" "$D/${s}_1_b.bin" Hypre 2>&1)
  t1=$(date +%s.%N)
  sub=$(echo "$out" | grep -c "Subnormal gamma")
  it=$(echo "$out"  | grep "num_iterations" | grep -oE "[0-9]+$" | tail -1)
  fr=$(echo "$out"  | grep "final_res_norm"  | grep -oE "[0-9.eE+-]+$" | tail -1)
  [ -z "$it" ] && { it="TIMEOUT"; fr="NA"; }
  # converged if final_res < 1e-10 and not bailed
  conv="?"
  if [ "$sub" -gt 0 ]; then conv="BAIL"
  elif [ "$it" = "TIMEOUT" ]; then conv="timeout"
  elif awk "BEGIN{exit !($fr < 1e-10)}"; then conv="CONVERGED"
  else conv="not-converged"; fi
  printf "  relax=%-2s np=%s step=%-3s | iters=%-6s final_res=%-11s wall=%5.1fs | %s\n" \
         "$rt" "$np" "$s" "$it" "${fr:-NA}" "$(awk "BEGIN{print $t1-$t0}")" "$conv"
}

echo "############################################################"
echo "# CONFIRM: relax=18 (l1-Jacobi) vs relax=8 (default) "
echo "#   on EXP-1's failing steps, np in {4,8}, full max_iter"
echo "############################################################"
for s in 20 40 50 71; do
  echo "-- step $s --"
  for np in 4 8; do
    run 8  $np $s     # default: expected BAIL
    run 18 $np $s     # proposed fix: expect CONVERGED (no bail)
  done
done
echo "ALL DONE"
