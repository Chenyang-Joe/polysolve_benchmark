#!/bin/bash
# Benchmark: Trilinos vs Hypre across different matrices and np values

EXE="build.trilinos_tpetra/TestMatLogger"
DATA_ROOT="/mnt/hdd1/chenyang/benchmark_data/larger_matrix_exp/new_mat_bin_support_large_index"
OUTFILE="benchmark_results.txt"

echo "============================================" > $OUTFILE
echo "Trilinos vs Hypre Benchmark" >> $OUTFILE
echo "Date: $(date)" >> $OUTFILE
echo "============================================" >> $OUTFILE

MATRICES=(
  "3D_golf_ball_39376_try_larger_matrix_0"
  "3D_golf_ball_73852_try_larger_matrix_0"
  "3D_golf_ball_113325_try_larger_matrix_0"
  "3D_golf_ball_189090_try_larger_matrix_0"
  "3D_golf_ball_vanilla_try_larger_matrix_0"
)

NP_VALUES=(1 2 4)
SOLVERS=("Trilinos" "Hypre")

for mat in "${MATRICES[@]}"; do
  A_FILE="$DATA_ROOT/$mat/1_1_A.bin"
  B_FILE="$DATA_ROOT/$mat/1_1_b.bin"

  if [ ! -f "$A_FILE" ] || [ ! -f "$B_FILE" ]; then
    echo "SKIP $mat: files not found" >> $OUTFILE
    continue
  fi

  echo "" >> $OUTFILE
  echo "--------------------------------------------" >> $OUTFILE
  echo "Matrix: $mat" >> $OUTFILE
  echo "--------------------------------------------" >> $OUTFILE

  for solver in "${SOLVERS[@]}"; do
    for np in "${NP_VALUES[@]}"; do
      echo "  Running: $solver np=$np $mat ..."

      # Run with timeout of 5 minutes
      result=$(timeout 300 mpirun --oversubscribe -np $np $EXE "$A_FILE" "$B_FILE" "$solver" 2>&1)
      exit_code=$?

      if [ $exit_code -ne 0 ]; then
        echo "  $solver | np=$np | FAILED (exit=$exit_code)" >> $OUTFILE
        continue
      fi

      # Extract metrics
      iters=$(echo "$result" | grep "num_iterations" | head -1 | grep -oP '[\d.e+-]+$')
      res_norm=$(echo "$result" | grep "final_res_norm" | head -1 | grep -oP '[\d.e+-]+$')
      residual=$(echo "$result" | grep "residual " | head -1 | grep -oP '[\d.e+-]+$')
      elapse=$(echo "$result" | grep "elapse_time" | head -1 | grep -oP '[\d.e+-]+(?=s)')
      factorize=$(echo "$result" | grep "factorize" | head -1 | grep -oP '[\d.e+-]+(?=s)')
      solve_t=$(echo "$result" | grep -P "solve \d" | head -1 | grep -oP '[\d.e+-]+(?=s)')

      printf "  %-9s | np=%-2s | iters=%-6s | res_norm=%-12s | residual=%-12s | factorize=%-8s | solve=%-8s | total=%-8s\n" \
        "$solver" "$np" "${iters:-N/A}" "${res_norm:-N/A}" "${residual:-N/A}" "${factorize:-N/A}" "${solve_t:-N/A}" "${elapse:-N/A}" >> $OUTFILE
    done
  done
done

echo "" >> $OUTFILE
echo "============================================" >> $OUTFILE
echo "Benchmark complete." >> $OUTFILE

cat $OUTFILE
