#include "../include/MatrixFlat.hpp"
#include "../include/mmm.hpp"

int main(){
    size_t dim = 5;

    MatrixFlat<float> A(dim, dim, -2, 2);
    MatrixFlat<float> B(dim, dim, -2, 2);
    MatrixFlat<float> C(dim, dim);


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

    auto dt = mmm_naive(A, B, C);
    C.print();

    std::cout<<"mmm with dimension: " <<dim<< " took: "<<dt<<" [ms]"<<std::endl;



    return 0;
}
