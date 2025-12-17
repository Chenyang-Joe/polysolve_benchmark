#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include <ctime>
#include <chrono>
#include <regex>


#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "polysolve/linear/Solver.hpp"
#include "save_problem.hpp"

#include <polysolve/Utils.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#if defined(SPDLOG_FMT_EXTERNAL)
#include <fmt/color.h>
#else
#include <spdlog/fmt/bundled/color.h>
#endif

// // Include MKL header for thread control
// #ifdef EIGEN_USE_MKL_ALL
// #include <mkl.h>
// #endif

// #include "polysolve/Utils.hpp"
// #include <tbb/global_control.h>

using namespace Eigen;
using namespace std;
using namespace benchy::io;
using namespace polysolve;




int main(int argc, char **argv)
{
    
    std::cout<<"[EXPBEGIN]"<<std::endl;
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
    // int num_threads = 1;  // default to 1 thread

    if (argc == 4)
    {
        solver_name = argv[3];  // "Hypre"
        is_nullspace = false;
    }
    else{
        nullspace_file = argv[3];
        solver_name = argv[4];  // "Hypre"
        is_nullspace = true;
    }


    // else if(argc == 5)
    // {
    // solver_name = argv[4];  // "Hypre"
    // if (solver_name.substr(0,5) != "Eigen")
    // {
	//     nullspace_file = argv[3];
    //     is_nullspace = true;
    // }else{
    //     //!!! this method does not work!!! do not use it!!! 
    //     // does not support nullspace and control threads together
    //     num_threads = std::stoi(argv[3]);  // Remove 'int' - use outer variable
    //     is_nullspace = false;
    //     // set number of threads for Eigen
    //     #ifdef EIGEN_USE_MKL_ALL
    //     mkl_set_dynamic(0); 
    //     mkl_set_num_threads(num_threads);
    //     std::cout << "MKL threads set to: " << mkl_get_max_threads() << std::endl;
    //     #endif
    //     Eigen::setNbThreads(num_threads);
    //     std::cout << "Eigen threads set to: " << Eigen::nbThreads() << std::endl;
    //     }
    // }
    
    
    // record time
    time_t begin_clock,end_clock;
    double ret;

    std::chrono::high_resolution_clock::time_point begin, end;
    double elapsed_seconds;


    // Load matrix
    polysolve::StiffnessMatrix A;
    int dim_local = 0;
    int is_symmetric_positive_definite = 0;
    int is_sequence_of_problems = 0;
    benchy::io::DeserializeStiffnessMatrix(A, dim_local, is_symmetric_positive_definite, is_sequence_of_problems, A_file);

    // printf("DESERIALIZING A\n");
    // std::cout << "DIM: " << dim_local <<  " IS_SPD: " << is_symmetric_positive_definite << " IS_SEQ: " << is_sequence_of_problems << std::endl;
    // std::cout << A << "\n" << std::endl;


    // define my logger
    static std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("test_logger");
    logger->set_level(spdlog::level::trace);    
    const static auto log_fmt_text_stats =
    fmt::format("[{}] {{}} {{:.5g}}", fmt::format(fmt::fg(fmt::terminal_color::magenta), "stats"));
    const static auto log_fmt_text_time =
    fmt::format("[{}] {{}} {{:.5g}}s", fmt::format(fmt::fg(fmt::terminal_color::magenta), "non_stopwatch_timing"));

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
    json input = {};
    input["solver"] = {solver_name};
    auto solver = linear::Solver::create(input, *logger);

    // // manually set parameters
    // if( solver_name == "AMGCL")
    // {    
    //         json params = {
    //     { "AMGCL", {
    //     { "solver", {
    //         { "tol", 1e-8 }       // <-- your new tolerance
    //     }}
    //     }}
    // };  
    //     solver->set_parameters(params);}
    // else if(solver_name == "Hypre"){

    //         json params = {
    //             { "Hypre", {
    //                 { "tolerance", 1e-8 }    // your new tolerance
    //             }}
    //         };
    //     solver->set_parameters(params);
    // }

    // Configuration parameters like iteration or accuracy for iterative solvers
    // solver->set_parameters(params);

    // // System sparse matrix
    // Eigen::SparseMatrix<double> A;

    // // Right-hand side
    // Eigen::VectorXd b;
    // Solution
    Eigen::VectorXd x(b.size());
    x.setZero();
    begin_clock=clock();
    begin = std::chrono::high_resolution_clock::now();

    solver->analyze_pattern(A, A.rows());
    solver->factorize(A);
    // if (is_nullspace == true)
    //     solver->solve(b, nullspace, x);
    // else
    //     solver->solve(b, x);
    solver->solve(b, x);
    end_clock=clock();
    end = std::chrono::high_resolution_clock::now();
    float residual = (A * x - b).norm(); // A x - b = 0
    logger->trace(log_fmt_text_stats, "residual", residual);


    bool direct_solver = false;
    if (solver_name ==  "AMGCL" || solver_name == "Hypre"){
        direct_solver =true;
    }


    json my_params = {};
    solver->get_info(my_params);
    double num_iter = 0.0;
    double final_res_norm = 0.0;
    double tol = 0.0;
    double maxiter = 0.0;
    if (direct_solver)
    {
    num_iter = my_params["num_iterations"];
    final_res_norm = my_params["final_res_norm"];
    tol = my_params["solver_tol"];
    maxiter = my_params["solver_maxiter"];
    }

    std::regex pattern(R"((\d+)_(\d+)_A\.bin)");
    std::smatch steps_match;
    double outer;
    double inner;
    if (std::regex_search(A_file, steps_match, pattern)) {
        outer = std::stod(steps_match[1]);  // 
        inner = std::stod(steps_match[2]); // 
    } else {
        outer = -1;  // 
        inner = -1; //    
    }
    logger->trace(log_fmt_text_stats, "outer", outer);
    logger->trace(log_fmt_text_stats, "inner", inner);

    if(direct_solver)
    {
        logger->trace(log_fmt_text_stats, "solver_tol", tol);
        logger->trace(log_fmt_text_stats, "solver_maxiter", maxiter);
        logger->trace(log_fmt_text_stats, "final_res_norm", final_res_norm);
        logger->trace(log_fmt_text_stats, "num_iterations", num_iter);
        logger->trace(log_fmt_text_stats, "norm_b", b.norm());
    }



    ret=(end_clock-begin_clock) / (double) CLOCKS_PER_SEC;
    elapsed_seconds = std::chrono::duration<double>(end - begin).count();
    // std::cout << "Solution: \n" << x << std::endl;

    // std::cout << "Condition Number: " << cond_number << std::endl;
    logger->trace(log_fmt_text_time, "clock_time", ret);
    logger->trace(log_fmt_text_time, "elapse_time", elapsed_seconds);
    // std::cout << "clock_time: " << ret << " ";
    // std::cout << "elapse_time: " << elapsed_seconds << " ";

    std::cout<<"[EXPEND]"<<std::endl;    
}
