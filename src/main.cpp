// src/main.cpp
#include <iostream>
#include "test.h"  // 包含subprogram.h头文件

#include "polysolve.h"
using namespace polysolve::linear;
using namespace std;

void printlist(){
    auto solvers = Solver::available_solvers();
    cout<< "A list of avaliable solvers" <<endl;
    for (const auto& solver : solvers) {
        cout << solver << endl;
    }

}


int main() {
    initializeMatrix();
    initializeVector();
    cout << "Direct Solver" <<endl;
    PardisoLU();
    PardisoLLT();
    PardisoLDLT();
    CholmodSupernodalLLT();
    SparseLU();
    trilinos();

    cout << "Iterative Solver" <<endl;
    // hypre();  // 调用subprogram.cpp中定义的函数
    ConjugateGradient();
    LeastSquaresConjugateGradient();

    // printlist();


}
