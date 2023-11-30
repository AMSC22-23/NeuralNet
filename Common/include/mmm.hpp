#include "MatrixFlat.hpp"

#ifndef MMM_HPP
#define MMM_HPP

// TO DO: All matrix A, B can be marked as const

int64_t mmm_blas(MatrixFlat<float>& A, MatrixFlat<float>& B, MatrixFlat<float>& C);

int64_t mmm_blas(MatrixFlat<double>& A, MatrixFlat<double>& B, MatrixFlat<double>& C);


int64_t mmm_naive(MatrixFlat<double>& A, MatrixFlat<double>& B, MatrixFlat<double>& C);

int64_t mmm_naive(MatrixFlat<float>& A, MatrixFlat<double>& B, MatrixFlat<float>& C);



#endif