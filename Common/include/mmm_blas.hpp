#include "MatrixFlat.hpp"
#include <cblas-openblas.h>
#include <chrono>

#ifndef MMM_BLAS_HPP
#define MMM_BLAS_HPP

void mmm_blas(MatrixFlat<float>& A, MatrixFlat<float>& B, MatrixFlat<float>& C, int64_t& time);

void mmm_blas(MatrixFlat<double>& A, MatrixFlat<double>& B, MatrixFlat<double>& C, int64_t& time);

#endif //MMM_BLAS_HPP