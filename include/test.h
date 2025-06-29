#ifndef TEST_H
#define TEST_H

#include <iostream>
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "polysolve.h"  // Assuming this header contains necessary declarations for Solver
#include <nlohmann/json.hpp>  // For JSON parameter handling

using namespace polysolve::linear;
using namespace std;

extern const int matrix_size;  // Declaring matrix size globally

// External variables
extern Eigen::SparseMatrix<double> A;
extern std::vector<Eigen::Triplet<double>> triplets;
extern Eigen::VectorXd b;

// Function declarations
void initializeMatrix();
void initializeVector();
void hypre();
void PardisoLU();
void PardisoLLT();
void PardisoLDLT();
void CholmodSupernodalLLT();
void SparseLU();
void ConjugateGradient();
void LeastSquaresConjugateGradient();
void trilinos();

#endif // TEST_H
