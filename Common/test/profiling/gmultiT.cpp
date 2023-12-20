/*
 * This script is needed by the class profiler to profile the algorithm mmm_gmultiT.
 * This program takes as input:
 * argv[1] = matrix dimensions
 * argv[2] = datatype: 0 for float, otherwise double
 * argv[3] = optimization flags, needed for id
 * argv[4] = tile dimension
 * argv[5] = the number of threads on which we want to run the algorithm
 * argv[6] = bool: true if we are running the script with valgrind - cachegrind
 *
 */


#include "../../include/mmm.hpp"
#include <string>
#include <fstream>

int main(int argc, char ** argv){

    if(argc != 7)
    {
        std::cout<<"Error! Wrong # of parameters"<<std::endl;
        return -1;
    }

    size_t dim = std::stoi(argv[1]);
    size_t T = std::stoi(argv[2]); // 0 = float  else = double
    std::string id = std::string (argv[3]);
    std::size_t tile_dim = std::stoi (argv[4]);
    int num_threads = std::stoi(argv[5]);
    bool cache_grind_run = std::stoi(argv[6]);

    int64_t time;

    if (T == 0) {
        std::cout<<"Float Version"<<std::endl;
        MatrixFlat<float> Af(dim, dim, -10, 10);
        MatrixFlat<float> Bf(dim, dim, -10, 10);
        MatrixFlat<float> Cf(dim, dim);

        mmm_gmultiT(Af, Bf, Cf, time, tile_dim, num_threads);

    }else {
        std::cout << "Double Version" << std::endl;
        MatrixFlat<double> A(dim, dim, -10, 10);
        MatrixFlat<double> B(dim, dim, -10, 10);
        MatrixFlat<double> C(dim, dim);
        mmm_gmultiT(A, B, C, time, tile_dim, num_threads);
    }

    if(!cache_grind_run) {
        std::string type = (T == 0) ? "float" : "double";
        std::string matrixDim = std::to_string(dim) + "X" + std::to_string(dim);
        appendCSVRow({"FM", id, matrixDim, type, std::to_string(time),
                      std::to_string(tile_dim), std::to_string(num_threads)});
    }

    return 0;
}