#include "../include/mmm.hpp"
/*
 *  To compile:
 *  g++ test_mmm_blas.cpp ../src/mmm.cpp -o TestMmmBlas -I ${mkOpenblasInc}  -L${mkOpenblasLib} -lopenblas
 */

int main(){
    std::size_t dim = 5;

    MatrixFlat<double> A(dim, dim, -2, 2);
    MatrixFlat<double> B(dim, dim, -2, 2);
    MatrixFlat<double> C(dim, dim);


    std::cout<<"Matrix A: "<<std::endl;
    A.print();

    std::cout<<"---------------------------------"<<std::endl;

    std::cout<<"Matrix B: "<<std::endl;

    B.print();

    std::cout<<"---------------------------------"<<std::endl;

    std::cout<<"Matrix C: "<<std::endl;

    C.print();

    std::cout<<"---------------------------------"<<std::endl;

    std::cout<<"Matrix C = A*B: "<<std::endl;

    auto dt = mmm_blas(A, B, C);
    C.print();

    std::cout<<"mmm with dimension: " <<dim<< " took: "<<dt<<" [ms]"<<std::endl;


    return 0;
}