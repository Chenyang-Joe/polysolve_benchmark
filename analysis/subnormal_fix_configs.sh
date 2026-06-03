#!/bin/bash
# Config comparison for analysis_2026-6-2-hypre-subnormal-fix.
# Metric: elapse_time (analyze+factorize+solve, rank0) + num_iterations.
# OMP_NUM_THREADS=1 (at np=8, OpenMPI default-binds 1 core/rank anyway -> matches 5-21).
BIN=/u/1/chenyang/benchmark/build.trilinos_tpetra/TestMatLogger
D=/mnt/hdd1/chenyang/benchmark_data/larger_matrix_exp/mat_twist/trial_1_result_part1/3146
S=71; TO=150
export OMP_NUM_THREADS=1
# args: np  PART PRECOND KRYLOV RELAX DIM NODEALIGN  label
run(){
  local np=$1 part=$2 pre=$3 kry=$4 rt=$5 dim=$6 na=$7 lbl="$8" out sub it fr et
  out=$(POLYSOLVE_HYPRE_PARTITION=$part POLYSOLVE_HYPRE_PRECOND=$pre POLYSOLVE_HYPRE_KRYLOV=$kry \
        POLYSOLVE_HYPRE_RELAX=$rt POLYSOLVE_HYPRE_DIM=$dim POLYSOLVE_HYPRE_NODE_ALIGNED=$na \
        timeout $TO mpirun -np $np $BIN $D/${S}_1_A.bin $D/${S}_1_b.bin Hypre 2>&1 </dev/null)
  sub=$(echo "$out"|grep -c "Subnormal gamma")
  it=$(echo "$out"|grep num_iterations|grep -oE "[0-9.]+$"|tail -1)
  fr=$(echo "$out"|grep final_res_norm|grep -oE "[0-9.eE+-]+$"|tail -1)
  et=$(echo "$out"|grep -oE "elapse_time [0-9.eE+-]+s"|grep -oE "[0-9.eE+-]+"|tail -1)
  local v
  if   [ "$sub" -gt 0 ]; then v="BAIL"
  elif [ -z "$it" ];     then v="TIMEOUT"
  elif awk "BEGIN{exit !($fr<1e-10)}"; then v="ok"
  else v="stuck"; fi
  printf "  np=%s | %-9s %-9s %-5s rlx=%-2s dim=%s na=%s | iter=%-6s time=%-8s res=%-10s | %s | %s\n" \
     "$np" "$part" "$pre" "$kry" "$rt" "$dim" "$na" "${it:-NA}" "${et:-NA}" "${fr:-NA}" "$v" "$lbl"
}

echo "########## SECTION 2: baseline (row_block + pcg + boomeramg, dim=1) relax sweep ##########"
for np in 1 8; do
  run $np row_block boomeramg pcg 8  1 0 "default-8"
  run $np row_block boomeramg pcg 16 1 0 "chebyshev-16"
  run $np row_block boomeramg pcg 18 1 0 "l1jacobi-18"
done

echo
echo "########## SECTION 3a: best PCG config search (dim=3 + node_aligned), np=8 ##########"
for part in row_block metis; do
  run 8 $part boomeramg pcg 16 3 1 "pcg amg cheby"
  run 8 $part boomeramg pcg 18 3 1 "pcg amg l1jac"
  run 8 $part euclid    pcg 18 3 1 "pcg euclid (relax NA)"
done

echo
echo "########## SECTION 3b: GMRES comparison (dim=3 + node_aligned + metis), np=8 ##########"
run 8 metis boomeramg gmres 8  3 1 "GMRES baseline (5-21 best)"
run 8 metis boomeramg gmres 16 3 1 "GMRES + cheby"
run 8 metis boomeramg gmres 18 3 1 "GMRES + l1jac"
echo "ALL DONE"
