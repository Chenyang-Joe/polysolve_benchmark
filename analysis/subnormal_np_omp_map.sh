#!/bin/bash
# (np x omp) interaction map for the hybrid-GS bail-out, step 71, default relax=8.
# Question: MPI ranks vs OpenMP threads -- which fragments the GS smoother enough
# to make M non-SPD?  Bail is an early event, so even oversubscribed runs report
# bail-or-not quickly; converging cases capped at TO.
set -u
BIN=/u/1/chenyang/benchmark/build.trilinos_tpetra/TestMatLogger
D=/mnt/hdd1/chenyang/benchmark_data/larger_matrix_exp/mat_twist/trial_1_result_part1/3146
S=71; TO=60

cell() { # relax np omp
  local rt=$1 np=$2 omp=$3 out sub it
  out=$(OMP_NUM_THREADS=$omp POLYSOLVE_HYPRE_RELAX=$rt timeout $TO \
        mpirun --oversubscribe -np "$np" "$BIN" "$D/${S}_1_A.bin" "$D/${S}_1_b.bin" Hypre 2>&1)
  sub=$(echo "$out" | grep -c "Subnormal gamma")
  it=$(echo "$out" | grep "num_iterations" | grep -oE "[0-9]+$" | tail -1)
  if [ "$sub" -gt 0 ]; then echo "BAIL"; elif [ -z "$it" ]; then echo "to"; else echo "ok($it)"; fi
}

echo "### relax=8 (default hybrid-GS): bail map over (np rows) x (omp cols), step 71 ###"
printf "%6s | %-8s %-8s %-8s\n" "np\\omp" "1" "16" "128"
for np in 1 2 3 4 8; do
  printf "%6s | %-8s %-8s %-8s\n" "$np" "$(cell 8 $np 1)" "$(cell 8 $np 16)" "$(cell 8 $np 128)"
done
echo
echo "### relax=18 (l1-Jacobi): same map -- expect ok everywhere ###"
printf "%6s | %-8s %-8s %-8s\n" "np\\omp" "1" "16" "128"
for np in 1 2 4 8; do
  printf "%6s | %-8s %-8s %-8s\n" "$np" "$(cell 18 $np 1)" "$(cell 18 $np 16)" "$(cell 18 $np 128)"
done
echo "ALL DONE"
