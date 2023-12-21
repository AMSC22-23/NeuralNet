#include "../../include/mmm_blas.hpp"
#include "../../include/mmm.hpp"
#include <string>
#include <fstream>

/*
 * The parameters of the program are
 *
 * argv[1] = matrix dimensions
 * argv[2] = datatype: 0 for float, otherwise double
 * argv[3] = optimization flags, needed for id
 * argv[4] = bool: true if we are running the script with valgrind - cachegrind
 */

int main(int argc, char ** argv){

    if(argc != 5)
    {
        std::cout<<"Error! Wrong # of parameters"<<std::endl;
        std::exit(-1);
    }

    size_t dim = std::stoi(argv[1]);
    size_t T = std::stoi(argv[2]); // 0 = float  else = double
    std::string id = std::string (argv[3]);
    bool cache_grind_run = std::stoi(argv[4]);


    std::cout<<"Matrix Dimension: "<<dim<<std::endl;
    int64_t time;



    if (T == 0) {
        std::cout<<"Float Version"<<std::endl;
        MatrixFlat<float> Af(dim, dim, -10, 10);
        MatrixFlat<float> Bf(dim, dim, -10, 10);
        MatrixFlat<float> Cf(dim, dim);

        mmm_blas(Af, Bf, Cf, time);

    }else {
        std::cout << "Double Version" << std::endl;
        MatrixFlat<double> A(dim, dim, -10, 10);
        MatrixFlat<double> B(dim, dim, -10, 10);
        MatrixFlat<double> C(dim, dim);
        mmm_blas(A, B, C, time);
    }

    if(!cache_grind_run) {
        std::string type = (T == 0) ? "float" : "double";
        std::string matrixDim = std::to_string(dim) + "X" + std::to_string(dim);
        std::string tile_size = "-1";  // -1 because we are not using tiles
        appendCSVRow({"FM", id, matrixDim, type, std::to_string(time),
                      "", "1"});
    }

    return 0;
}