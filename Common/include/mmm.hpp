#include "MatrixFlat.hpp"

#ifndef MMM_HPP
#define MMM_HPP


void mmm_blas(MatrixFlat<float>& A, MatrixFlat<float>& B, MatrixFlat<float>& C, int64_t& time);

void mmm_blas(MatrixFlat<double>& A, MatrixFlat<double>& B, MatrixFlat<double>& C, int64_t& time);

void mmm_naive(const MatrixFlat<double>& A, const MatrixFlat<double>& B, MatrixFlat<double>& C, int64_t& time);

void mmm_naive(const MatrixFlat<float>& A, const MatrixFlat<float>& B, MatrixFlat<float>& C, int64_t& time);

void mmm_naive_RegisterAcc(const MatrixFlat<double>& A, const MatrixFlat<double>& B, MatrixFlat<double>& C, int64_t& time);

void mmm_naive_RegisterAcc(const MatrixFlat<float>& A, const MatrixFlat<float>& B, MatrixFlat<float>& C, int64_t& time);

void mmm_loopI(const MatrixFlat<double>& A, const MatrixFlat<double>& B, MatrixFlat<double>& C, int64_t& time);

void mmm_loopI(const MatrixFlat<float>& A, const MatrixFlat<float>& B, MatrixFlat<float>& C, int64_t& time);

void mmm_tiling(const MatrixFlat<double>& A, const MatrixFlat<double>& B, MatrixFlat<double>& C, int64_t& time, int tileSize);

void mmm_tiling(const MatrixFlat<float>& A, const MatrixFlat<float>& B, MatrixFlat<float>& C, int64_t& time, int tileSize);

void printFile(std::ofstream& file, int id, size_t m, size_t n, int T, int64_t& time);

#endif