#include "Matrix.hpp"
#include<iostream>  


int main(){


    Matrix<double> A(5, 5, false); 

    std::cout<<"Matrix is in storage mode: " << A.get_storage_type() <<std::endl; 
    A.random_fill(-300., 300., 42); 
    A.print();

    return 0; 
}