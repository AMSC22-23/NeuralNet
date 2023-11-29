//
// Created by filippo on 29/11/23.
//
# include "../include/MatrixFlat.hpp"




int main(){

    size_t rows = 4;
    size_t cols = 5;

    //MatrixFlat<double> A(rows, cols, -2, 2);

    MatrixFlat<double> A(rows, cols);
    A.print();

    return 0;
}