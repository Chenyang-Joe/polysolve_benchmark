#!/bin/bash
# Hypre v3.1.0 regression + relax_type=88 experiment for
# analysis_2026-6-2-hypre-subnormal-fix.  Same recipe / metric as
# analysis/subnormal_fix_configs.sh, but BIN points at the v3.1.0 build.
BIN=/u/1/chenyang/benchmark/build.hypre_latest/TestMatLogger
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

echo "########## REGRESSION on hypre v3.1.0: do the v2.28.0 configs still work? ##########"
for np in 1 8; do
  run $np row_block boomeramg pcg   8  3 1 "baseline (row_block+pcg+amg+GS8) -- expect np1 ok, np8 BAIL"
  run $np metis     boomeramg gmres 8  3 1 "formal best (metis+gmres+amg+GS8)"
  run $np metis     boomeramg pcg   18 3 1 "current best PCG (metis+pcg+amg+l1jac18)"
done

echo
echo "########## NEW: relax_type=88 (convergent-l1 hybrid SSOR) + pcg + boomeramg + dim3 + node-aligned ##########"
for np in 1 8; do
  run $np row_block boomeramg pcg 88 3 1 "relax88 row_block pcg"
  run $np metis     boomeramg pcg 88 3 1 "relax88 metis pcg"
done
echo "ALL DONE"
