#include "../include/mmm.hpp"
#include <string>

int main(int argc, char ** argv){

    if(argc != 2)
    {
        std::cout<<"Error! You must pass just one positive value to the program. "<<std::endl;
        std::exit(-1);
    }

    size_t dim = std::stoi(argv[1]);

    MatrixFlat<double> A(dim, dim, -10, 10);
    MatrixFlat<double> B(dim, dim, -10, 10);
    MatrixFlat<double> C(dim, dim);
    MatrixFlat<float> Af(dim, dim, -10, 10);
    MatrixFlat<float> Bf(dim, dim, -10, 10);
    MatrixFlat<float> Cf(dim, dim);


    int64_t time;

    mmm_naive(A, B, C, time);
    std::cout<<"This operation took: "<<time<< " [ms]"<<std::endl;
    mmm_blas(A, B, C, time);
    std::cout<<"The same operation using openBlas took: "<<time<< " [ms]"<<std::endl;
    std::cout<<"-----------------------------------------------------------------------"<<std::endl;

    mmm_naive(Af, Bf, Cf, time);
    std::cout<<"This operation took: "<<time<< " [ms]"<<std::endl;
    mmm_blas(Af, Bf, Cf, time);
    std::cout<<"The same operation using openBlas took: "<<time<< " [ms]"<<std::endl;
    std::cout<<"-----------------------------------------------------------------------"<<std::endl;



}