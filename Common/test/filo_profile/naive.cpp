#include "../../include/mmm.hpp"
#include <string>

/*
 * The parameters of the program are
 *
 * argv[1] = matrix dimensions
 * argv[2] = datatype: 0 for float, otherwise double
 * argv[3] = output filename
 * argv[4] = optimization flag
 */

int main(int argc, char ** argv){

    if(argc != 5)
    {
        std::cout<<"Error! Wrong # of parameters"<<std::endl;
        std::exit(-1);
    }

    size_t dim = std::stoi(argv[1]);
    size_t T = std::stoi(argv[2]); // 0 = float  else = double
    std::string filename = argv[3];
    std::string id = std::string (argv[4]) + "1";
    std::cout<<"Input Dimension: "<<dim<<std::endl;
    std::cout<<"Matrices will be of dimensions: "<<dim<<"X"<<dim<<std::endl;
    int64_t time;



    if (T == 0) {
        MatrixFlat<float> Af(dim, dim, -10, 10);
        MatrixFlat<float> Bf(dim, dim, -10, 10);
        MatrixFlat<float> Cf(dim, dim);

        mmm_naive(Af, Bf, Cf, time);
        printFile(filename, id, dim, dim, T, time);
        return 0;
    }

    MatrixFlat<double> A(dim, dim, -10, 10);
    MatrixFlat<double> B(dim, dim, -10, 10);
    MatrixFlat<double> C(dim, dim);
    mmm_naive(A, B, C, time);
    printFile();


}