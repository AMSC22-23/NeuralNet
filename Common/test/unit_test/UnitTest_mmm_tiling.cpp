#include "../../include/mmm.hpp"
#include "../../include/mmm_blas.hpp"

/*
 * This test has the scope of validate the mmm_tiling algorithm.
 * We test if the function works, in both double & single precision, we compare the result with the openBlas
 * matrix-matrix multiplication in both term of times and correctness of result.
 *
 * To compile (with -O3 -march=native -ffast-math) :
 * make UnitTest_mmm_tiling
 *
 * To run this test you have to pass the desired dimension of the matrix and the dimension of tile to be loaded in CPU cache
 *
 */


int main(int argc, char ** argv){

    if(argc != 3)
    {
        std::cout<<"Error! You must pass two positive values to the program. "<<std::endl;
        std::exit(-1);
    }

    size_t dim = std::stoi(argv[1]);
    size_t tileSize = std::stoi(argv[2]);

    if(dim % tileSize != 0)
    {
        std::cout<<"Error! The dimension of the matrix must be a multiple of the tileSize"<<std::endl;
        std::exit(-1);
    }

    std::cout<<"Input Dimension: "<<dim<<std::endl;
    std::cout<<"Matrices will be of dimensions: "<<dim<<"X"<<dim<<std::endl;

    MatrixFlat<double> A(dim, dim, -10, 10);
    MatrixFlat<double> B(dim, dim, -10, 10);
    MatrixFlat<double> C(dim, dim);
    MatrixFlat<double> Cblas(dim, dim);
    MatrixFlat<float> Af(dim, dim, -10, 10);
    MatrixFlat<float> Bf(dim, dim, -10, 10);
    MatrixFlat<float> Cf(dim, dim);
    MatrixFlat<float> Cblasf(dim, dim);


    int64_t time;

    mmm_tiling(A, B, C, time, tileSize);
    std::cout<<"This operation took: "<<time<< " [ms]"<<std::endl;
    mmm_blas(A, B, Cblas, time);
    std::cout<<"The same operation using openBlas took: "<<time<< " [ms]"<<std::endl;
    std::cout<<"We check if the result is the same: "<<std::endl;
    std::cout<<"nnz(C-Cblas): "<<(Cblas-C).nnzrs()<<std::endl;

    std::cout<<"-----------------------------------------------------------------------"<<std::endl;

    mmm_tiling(Af, Bf, Cf, time, tileSize);
    std::cout<<"This operation took: "<<time<< " [ms]"<<std::endl;
    mmm_blas(Af, Bf, Cblasf, time);
    std::cout<<"The same operation using openBlas took: "<<time<< " [ms]"<<std::endl;
    std::cout<<"We check if the result is the same: "<<std::endl;
    std::cout<<"nnz(C-Cblas): "<<(Cblas-C).nnzrs()<<std::endl;

    std::cout<<"-----------------------------------------------------------------------"<<std::endl;


}
