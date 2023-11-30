#include "MatrixFlat.hpp"

#ifndef MMM_HPP
#define MMM_HPP


void mmm_blas(MatrixFlat<float>& A, MatrixFlat<float>& B, MatrixFlat<float>& C, int64_t& time);

void mmm_blas(MatrixFlat<double>& A, MatrixFlat<double>& B, MatrixFlat<double>& C, int64_t& time);

void mmm_naive(const MatrixFlat<double>& A, const MatrixFlat<double>& B, MatrixFlat<double>& C, int64_t& time);

void mmm_naive(const MatrixFlat<float>& A, const MatrixFlat<float>& B, MatrixFlat<float>& C, int64_t& time);




#endif