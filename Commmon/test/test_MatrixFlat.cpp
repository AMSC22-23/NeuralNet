#include"../include/MatrixFlat.hpp"
#include <iostream>
int main(){


    // Testing Random Constructor of MatrixFlat
    MatrixFlat<double> A(4, 4, -10, 10);


    std::cout<<"Testing random constructor: "<<std::endl;
    A._print(std::cout); // _print & operator() const Testing;
    std::cout<<"-----------------------------------------------------"<<std::endl;


    std::cout<<" Testing Zero constructor of A: "<<std::endl;
    MatrixFlat<float> B(4, 4);

    std::cout<<"Testing print() : "<<std::endl;
    B.print() ;

    std::cout<<"-----------------------------------------------------"<<std::endl;
    auto ptr = B.get_ptr(); //Testing get_ptr

    std::cout<<"Modifing B matrix in an unsafe way to test if value returned by get_ptr is correct: "<<std::endl;
    for (int i = 0; i< 16; i++) {
        *ptr = 1.;
        ptr ++;
    }

    B._print(std::cout);



    std::cout<<"-----------------------------------------------------"<<std::endl;

    std::cout<<"Testing operator () :"<<std::endl;
    B(3, 3) = 7;

    B.print(std::cout);

    B(12, 12) = 12;

    B.print();


    return 0; 
}