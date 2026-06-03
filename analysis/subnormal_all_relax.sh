#!/bin/bash
# Exhaustive test of EVERY implemented BoomerAMG relax_type as a CG-preconditioner
# smoother. Canonical hard matrix mat_twist/3146 step 71.
#   Phase 1: np=4, omp=1  (pure MPI fragmentation - the 5-21 failure regime)
#   Phase 2: survivors also tested at np=1, omp=16 (pure THREAD fragmentation)
# Verdict: BAIL=non-SPD trap | CONVERGED(it) | stuck(it) | to(timeout,SPD) | ERR
BIN=/u/1/chenyang/benchmark/build.trilinos_tpetra/TestMatLogger
D=/mnt/hdd1/chenyang/benchmark_data/larger_matrix_exp/mat_twist/trial_1_result_part1/3146
S=71; TO=70
declare -A NAME=(
 [0]="Weighted Jacobi" [1]="seq Gauss-Seidel (VERY SLOW)" [2]="GS interior-par/bndry-seq"
 [3]="hybrid GS forward" [4]="hybrid GS backward" [5]="hybrid CHAOTIC GS"
 [6]="hybrid symmetric GS/SSOR" [7]="Jacobi (matvec)" [8]="l1 hybrid symm GS (default)"
 [9]="Gaussian elim (coarse-only)" [10]="case10" [11]="two-stage GS fwd"
 [12]="two-stage GS (diag)" [13]="l1 GS forward" [14]="l1 GS backward"
 [15]="CG smoother (variable!)" [16]="Chebyshev" [17]="FCF-Jacobi"
 [18]="l1 Jacobi" [19]="direct GE" [20]="Kaczmarz" [98]="direct GE+BLAS pivot")
run(){ local rt=$1 np=$2 omp=$3 out sub it fr
  out=$(OMP_NUM_THREADS=$omp POLYSOLVE_HYPRE_RELAX=$rt timeout $TO mpirun --oversubscribe -np $np $BIN $D/${S}_1_A.bin $D/${S}_1_b.bin Hypre 2>&1)
  sub=$(echo "$out"|grep -c "Subnormal gamma"); it=$(echo "$out"|grep num_iter|grep -oE "[0-9]+$"|tail -1); fr=$(echo "$out"|grep final_res|grep -oE "[0-9.eE+-]+$"|tail -1)
  if   [ "$sub" -gt 0 ]; then echo "BAIL";
  elif [ -z "$it" ];     then echo "to";
  elif awk "BEGIN{exit !($fr<1e-10)}"; then echo "CONVERGED($it)";
  else echo "stuck($it,res=$fr)"; fi; }
ORDER="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 98"
echo "===== Phase 1: np=4 omp=1 (MPI fragmentation) ====="
for rt in $ORDER; do
  v=$(run $rt 4 1); printf "  relax=%-3s %-32s -> %s\n" "$rt" "${NAME[$rt]}" "$v"
  echo "$rt $v" >> /tmp/relax_phase1.tmp
done
echo
echo "===== Phase 2: survivors of phase1, at np=1 omp=16 (THREAD fragmentation) ====="
while read rt v; do
  case "$v" in CONVERGED*) v2=$(run $rt 1 16); printf "  relax=%-3s %-32s -> %s\n" "$rt" "${NAME[$rt]}" "$v2";; esac
done < /tmp/relax_phase1.tmp
echo "ALL DONE"
