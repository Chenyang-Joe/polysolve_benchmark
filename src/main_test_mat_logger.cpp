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
    time_t begin_clock,end_clock;
    double ret;

    std::chrono::high_resolution_clock::time_point begin, end;
    double elapsed_seconds;


    // Load matrix
    Eigen::SparseMatrix<double> A;
    int dim_local = 0;
    int is_symmetric_positive_definite = 0;
    int is_sequence_of_problems = 0;
    Deserialize(A, dim_local, is_symmetric_positive_definite, is_sequence_of_problems, A_file);

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

    // Configuration parameters like iteration or accuracy for iterative solvers
    // solver->set_parameters(params);

    // // System sparse matrix
    // Eigen::SparseMatrix<double> A;

    // // Right-hand side
    // Eigen::VectorXd b;
    // Solution
    Eigen::VectorXd x(b.size());

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


    json my_params = {};
    solver->get_info(my_params);
    double num_iter = my_params["num_iterations"];
    double final_res_norm = my_params["final_res_norm"];
    double tol = my_params["solver_tol"];
    double maxiter = my_params["solver_maxiter"];

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

    logger->trace(log_fmt_text_stats, "solver_tol", tol);
    logger->trace(log_fmt_text_stats, "solver_maxiter", maxiter);
    logger->trace(log_fmt_text_stats, "final_res_norm", final_res_norm);
    logger->trace(log_fmt_text_stats, "num_iterations", num_iter);
    logger->trace(log_fmt_text_stats, "norm_b", b.norm());


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
