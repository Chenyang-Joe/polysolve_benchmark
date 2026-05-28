// bin_to_mtx.cpp
//
// Convert a PolyFEM-dumped (A.bin, b.bin) problem to Matrix Market (.mtx).
//
// It loads the pair exactly the way src/main_test_mat_logger.cpp does
//   - A : benchy::io::DeserializeStiffnessMatrix  (sparse, handles the int64
//         "LARGE_INDEX" format with the -1 marker as well as legacy int32)
//   - b : benchy::io::ReadMat                     (dense rows,cols + raw data)
// then writes them back out with Eigen's Matrix Market writers:
//   - A -> <prefix>_A.mtx   via Eigen::saveMarket        (sparse coordinate)
//   - b -> <prefix>_b.mtx   via Eigen::saveMarketVector  (dense vector)
//
// Usage:
//   BinToMtx <A.bin> <b.bin> [out_prefix]
// If out_prefix is omitted it is derived by stripping the trailing "_A.bin"
// from the A file, so .../71_1_A.bin -> .../71_1_A.mtx and .../71_1_b.mtx.
//
// Build: a `BinToMtx` target is wired into the top-level CMakeLists.txt next to
// TestMatLogger (same polysolve link, so POLYSOLVE_LARGE_INDEX matches).

#include <iostream>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

#include "polysolve/Types.hpp"
#include "save_problem.hpp"

using namespace benchy::io;

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <A.bin> <b.bin> [out_prefix]\n"
                  << "  writes <out_prefix>_A.mtx and <out_prefix>_b.mtx\n"
                  << "  default out_prefix strips trailing \"_A.bin\" from <A.bin>\n";
        return 1;
    }

    const std::string A_file = argv[1];
    const std::string b_file = argv[2];

    // ---- derive output prefix ----
    std::string prefix;
    if (argc >= 4)
    {
        prefix = argv[3];
    }
    else
    {
        prefix = A_file;
        const std::string suf = "_A.bin";
        if (prefix.size() >= suf.size()
            && prefix.compare(prefix.size() - suf.size(), suf.size(), suf) == 0)
            prefix.erase(prefix.size() - suf.size());
    }
    const std::string outA = prefix + "_A.mtx";
    const std::string outb = prefix + "_b.mtx";

    // ---- load A, exactly like main_test_mat_logger.cpp ----
    polysolve::StiffnessMatrix A;
    int dim = 0, is_spd = 0, is_seq = 0;
    DeserializeStiffnessMatrix(A, dim, is_spd, is_seq, A_file);
    std::cout << "loaded A: " << A.rows() << " x " << A.cols()
              << "  nnz=" << A.nonZeros()
              << "  (dim=" << dim << " is_spd=" << is_spd
              << " is_seq=" << is_seq << ")\n";

    // ---- load b, exactly like main_test_mat_logger.cpp ----
    Eigen::MatrixXd b;
    ReadMat(b, b_file);
    std::cout << "loaded b: " << b.rows() << " x " << b.cols() << "\n";

    if (b.rows() != A.rows())
        std::cerr << "WARNING: b.rows()=" << b.rows()
                  << " != A.rows()=" << A.rows() << "\n";

    // ---- write Matrix Market ----
    if (!Eigen::saveMarket(A, outA))
    {
        std::cerr << "ERROR: saveMarket failed for " << outA << "\n";
        return 2;
    }

    // b is stored as a dense (N x 1) matrix; saveMarketVector wants a vector
    Eigen::VectorXd bv = b.col(0);
    if (!Eigen::saveMarketVector(bv, outb))
    {
        std::cerr << "ERROR: saveMarketVector failed for " << outb << "\n";
        return 3;
    }

    std::cout << "wrote " << outA << "\n"
              << "wrote " << outb << "\n";
    return 0;
}
