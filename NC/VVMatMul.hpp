#include "MatrixVect2.hpp"

#ifndef VVMATMUL_HPP
#define VVMATMUL_HPP

template<typename T>
MatrixVect<T> VVMatMulNaive(const MatrixVect<T>& matrix1, const MatrixVect<T>& matrix2);

template<typename T>
MatrixVect<T> VVMatMulNaive2(const MatrixVect<T>& matrix1, const MatrixVect<T>& matrix2);

template<typename T>
MatrixVect<T> VVMatMulCacheFriendly(const MatrixVect<T>& matrix1, const MatrixVect<T>& matrix2);

#endif
