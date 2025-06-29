// src/subprogram.cpp
#include "test.h"  

using namespace polysolve::linear;
using namespace std;

const int matrix_size = 4;

// Create Sparse Matrix A globally
Eigen::SparseMatrix<double> A(matrix_size, matrix_size);
std::vector<Eigen::Triplet<double>> triplets;

void initializeMatrix() {
    triplets.push_back(Eigen::Triplet<double>(0, 0, 1.0));
    triplets.push_back(Eigen::Triplet<double>(1, 1, 2.0));
    triplets.push_back(Eigen::Triplet<double>(2, 2, 3.0));
    triplets.push_back(Eigen::Triplet<double>(3, 3, 4.0));
    A.setFromTriplets(triplets.begin(), triplets.end());
}

// Right-hand side vector b globally
Eigen::VectorXd b(matrix_size);

void initializeVector() {
    b << 1, 2, 3, 4;
}

void hypre(){

    
    Eigen::VectorXd x(matrix_size);
    
    //(make sure to turn on Hypre at CMake setting)
    const std::string solver_name = "Hypre";
    auto solver = Solver::create(solver_name, "");
    
    // setting parameters if required
    // solver->set_parameters(params);

    nlohmann::json params;
    params["max_iterations"] = 1000;  // Example parameter
    params["tolerance"] = 1e-6;      // Example tolerance for convergence
    solver->set_parameters(params);

    
    solver->analyze_pattern(A, A.rows());
    solver->factorize(A);
    solver->solve(b, x);
    
    cout << "hypre" <<endl;
    cout << "Solving x: " << x.transpose() << endl;

}


void trilinos(){


    Eigen::VectorXd x(matrix_size);
    
    //(make sure to turn on Hypre at CMake setting)
    const std::string solver_name = "Trilinos";
    cout << "hello" <<endl;

    auto solver = Solver::create(solver_name, "");
    cout << "bye" <<endl;


    // setting parameters if required
    // solver->set_parameters(params);

    // nlohmann::json params;
    // params["max_iterations"] = 1000;  // Example parameter
    // params["tolerance"] = 1e-6;      // Example tolerance for convergence
    // solver->set_parameters(params);

    cout << "trilinos starts" <<endl;

    solver->analyze_pattern(A, A.rows());
    cout << "start factorize" <<endl;

    solver->factorize(A);
    cout << "start solve" <<endl;

    solver->solve(b, x);
    cout << "end solve" <<endl;

    
    cout << "trilinos" <<endl;
    cout << "Solving x: " << x.transpose() << endl;

}





void PardisoLU(){
    Eigen::VectorXd x(matrix_size);
    
    //(make sure to turn on Hypre at CMake setting)
    const std::string solver_name = "Eigen::PardisoLU";
    auto solver = Solver::create(solver_name, "");
    
    // setting parameters if required
    // solver->set_parameters(params);
    
    solver->analyze_pattern(A,A.rows());
    solver->factorize(A);
    solver->solve(b, x);
    
    cout << "PardisoLU" <<endl;
    cout << "Solving x: " << x.transpose() << endl;

}


void PardisoLLT(){
    Eigen::VectorXd x(matrix_size);
    
    //(make sure to turn on Hypre at CMake setting)
    const std::string solver_name = "Eigen::PardisoLLT";
    auto solver = Solver::create(solver_name, "");
    
    // setting parameters if required
    // solver->set_parameters(params);
    
    solver->analyze_pattern(A,A.rows());
    solver->factorize(A);
    solver->solve(b, x);
    
    cout << "PardisoLLT" <<endl;
    cout << "Solving x: " << x.transpose() << endl;

}

void PardisoLDLT(){
    Eigen::VectorXd x(matrix_size);
    
    //(make sure to turn on Hypre at CMake setting)
    const std::string solver_name = "Eigen::PardisoLDLT";
    auto solver = Solver::create(solver_name, "");
    
    // setting parameters if required
    // solver->set_parameters(params);
    
    solver->analyze_pattern(A,A.rows());
    solver->factorize(A);
    solver->solve(b, x);
    
    cout << "PardisoLDLT" <<endl;
    cout << "Solving x: " << x.transpose() << endl;

}

void CholmodSupernodalLLT(){
    Eigen::VectorXd x(matrix_size);
    
    //(make sure to turn on Hypre at CMake setting)
    const std::string solver_name = "Eigen::CholmodSupernodalLLT";
    auto solver = Solver::create(solver_name, "");
    
    // setting parameters if required
    // solver->set_parameters(params);
    
    solver->analyze_pattern(A,A.rows());
    solver->factorize(A);
    solver->solve(b, x);
    
    cout << "CholmodSupernodalLLT" <<endl;
    cout << "Solving x: " << x.transpose() << endl;

}

void SparseLU(){
    Eigen::VectorXd x(matrix_size);
    
    //(make sure to turn on Hypre at CMake setting)
    const std::string solver_name = "Eigen::SparseLU";
    auto solver = Solver::create(solver_name, "");
    
    // setting parameters if required
    // solver->set_parameters(params);
    
    solver->analyze_pattern(A,A.rows());
    solver->factorize(A);
    solver->solve(b, x);
    
    cout << "SparseLU" <<endl;
    cout << "Solving x: " << x.transpose() << endl;

}


void ConjugateGradient(){
    Eigen::VectorXd x(matrix_size);
    
    //(make sure to turn on Hypre at CMake setting)
    const std::string solver_name = "Eigen::ConjugateGradient";
    auto solver = Solver::create(solver_name, "");
    
    // setting parameters if required
    // solver->set_parameters(params);
    
    solver->analyze_pattern(A,A.rows());
    solver->factorize(A);
    solver->solve(b, x);
    
    cout << "ConjugateGradient" <<endl;
    cout << "Solving x: " << x.transpose() << endl;

}

void LeastSquaresConjugateGradient(){
    Eigen::VectorXd x(matrix_size);
    
    //(make sure to turn on Hypre at CMake setting)
    const std::string solver_name = "Eigen::LeastSquaresConjugateGradient";
    auto solver = Solver::create(solver_name, "");
    
    // setting parameters if required
    // solver->set_parameters(params);
    
    solver->analyze_pattern(A,A.rows());
    solver->factorize(A);
    solver->solve(b, x);
    
    cout << "LeastSquaresConjugateGradient" <<endl;
    cout << "Solving x: " << x.transpose() << endl;

}