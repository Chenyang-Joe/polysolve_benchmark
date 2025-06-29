#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include <ctime>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "polysolve/linear/Solver.hpp"
#include "save_problem.hpp"

using namespace Eigen;
using namespace std;
using namespace benchy::io;
using namespace polysolve;


int main(int argc, char **argv)
{
    if (argc < 4) 
    {
        std::cout << "Missing args: matrix A file, matrix b file, nullspace file(optional), solver name" << std::endl;
        return 1;
    }

    std::string A_file = argv[1];
	std::string b_file = argv[2];
    std::string nullspace_file;
    std::string solver_name;
    bool is_nullspace;

    if (argc == 4)
    {
        solver_name = argv[3];  // "Hypre"
        is_nullspace = false;
    }
    else
    {
	    nullspace_file = argv[3];
        solver_name = argv[4];  // "Hypre"
        is_nullspace = true;
    }
    
    // record time
    time_t begin,end;
    double ret;

    // Load matrix
    Eigen::SparseMatrix<double> A;
    int dim_local = 0;
    int is_symmetric_positive_definite = 0;
    int is_sequence_of_problems = 0;
    Deserialize(A, dim_local, is_symmetric_positive_definite, is_sequence_of_problems, A_file);

    printf("DESERIALIZING A\n");
    std::cout << "DIM: " << dim_local <<  " IS_SPD: " << is_symmetric_positive_definite << " IS_SEQ: " << is_sequence_of_problems << std::endl;
    // std::cout << A << "\n" << std::endl;

    Eigen::MatrixXd b;
    ReadMat(b, b_file);

    // printf("DESERIALIZING b\n");
    // std::cout << b << "\n" << std::endl;

    Eigen::MatrixXd nullspace;
    if (is_nullspace == true)
    {
        ReadMat(nullspace, nullspace_file);
        // printf("DESERIALIZING nullspace\n");
        // std::cout << nullspace << "\n" << std::endl;
    }

    // Solve
    auto solver = linear::Solver::create(solver_name, "");

    // Configuration parameters like iteration or accuracy for iterative solvers
    // solver->set_parameters(params);

    // // System sparse matrix
    // Eigen::SparseMatrix<double> A;

    // // Right-hand side
    // Eigen::VectorXd b;

    // Solution
    Eigen::VectorXd x(b.size());

    begin=clock();
    solver->analyze_pattern(A, A.rows());
    solver->factorize(A);
    // if (is_nullspace == true)
    //     solver->solve(b, nullspace, x);
    // else
    //     solver->solve(b, x);
    solver->solve(b, x);
    end=clock();
    
    ret=(end-begin) / (double) CLOCKS_PER_SEC;
    // std::cout << "Solution: \n" << x << std::endl;
    std::cout << "[BOS]" << ret << "[EOS]";

}
