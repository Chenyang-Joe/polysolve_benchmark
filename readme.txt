TestMat matrix A file, matrix b file, nullspace file(optional), solver name



./TestMat /u/1/chenyang/matrix_resource/solver-mat-0906/2-cubes/1_1_A.bin /u/1/chenyang/matrix_resource/solver-mat-0906/2-cubes/1_1_b.bin trilinos

To check 
export OMP_NUM_THREADS=1
./TestMatTime /u/1/chenyang/matrix_resource/solver-mat-0906/golf-ball-doformable-wall/59_2_A.bin /u/1/chenyang/matrix_resource/solver-mat-0906/golf-ball-doformable-wall/59_2_b.bin AMGCL
./TestMatTime /u/1/chenyang/matrix_resource/solver-mat-0906/golf-ball-doformable-wall/74_2_A.bin /u/1/chenyang/matrix_resource/solver-mat-0906/golf-ball-doformable-wall/74_2_b.bin AMGCL
export OMP_NUM_THREADS=1
./TestMatTime /u/1/chenyang/matrix_resource/solver-mat-0906/golf-ball-doformable-wall/3_2_A.bin /u/1/chenyang/matrix_resource/solver-mat-0906/golf-ball-doformable-wall/3_2_b.bin AMGCL
./TestMatTime /u/1/chenyang/matrix_resource/solver-mat-0906/golf-ball-doformable-wall/70_1_A.bin /u/1/chenyang/matrix_resource/solver-mat-0906/golf-ball-doformable-wall/70_1_b.bin AMGCL


./TestMatTime /u/1/chenyang/matrix_resource/solver-mat-0906/golf-ball-doformable-wall/75_4_A.bin /u/1/chenyang/matrix_resource/solver-mat-0906/golf-ball-doformable-wall/75_4_b.bin Hypre



./TestMatCN /u/1/chenyang/matrix_resource/solver-mat-0906/disk-codim-points/19_28_A.bin /u/1/chenyang/matrix_resource/solver-mat-0906/disk-codim-points/19_28_b.bin Hypre


./TestMatLogger /u/1/chenyang/matrix_resource/solver-mat-0906/golf-ball-doformable-wall/59_2_A.bin /u/1/chenyang/matrix_resource/solver-mat-0906/golf-ball-doformable-wall/59_2_b.bin AMGCL

./TestMatLogger /u/1/chenyang/matrix_resource/solver-mat-0906/golf-ball-doformable-wall/74_2_A.bin /u/1/chenyang/matrix_resource/solver-mat-0906/golf-ball-doformable-wall/74_2_b.bin AMGCL


cmake -DCMAKE_BUILD_TYPE=Debug ..
gdb ./TestMatLogger
run /u/1/chenyang/matrix_resource/solver-mat-0906/golf-ball-doformable-wall/59_2_A.bin /u/1/chenyang/matrix_resource/solver-mat-0906/golf-ball-doformable-wall/59_2_b.bin AMGCL
