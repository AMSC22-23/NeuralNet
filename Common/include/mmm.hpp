#include "MatrixFlat.hpp"
#include <string>

#ifndef MMM_HPP
#define MMM_HPP



void mmm_naive(const MatrixFlat<double>& A, const MatrixFlat<double>& B, MatrixFlat<double>& C, int64_t& time);

void mmm_naive(const MatrixFlat<float>& A, const MatrixFlat<float>& B, MatrixFlat<float>& C, int64_t& time);

void mmm_naive_RegisterAcc(const MatrixFlat<double>& A, const MatrixFlat<double>& B, MatrixFlat<double>& C, int64_t& time);

void mmm_naive_RegisterAcc(const MatrixFlat<float>& A, const MatrixFlat<float>& B, MatrixFlat<float>& C, int64_t& time);

void mmm_loopI(const MatrixFlat<double>& A, const MatrixFlat<double>& B, MatrixFlat<double>& C, int64_t& time);

void mmm_loopI(const MatrixFlat<float>& A, const MatrixFlat<float>& B, MatrixFlat<float>& C, int64_t& time);

void mmm_tiling(const MatrixFlat<double>& A, const MatrixFlat<double>& B, MatrixFlat<double>& C, int64_t& time, int tileSize);

void mmm_tiling(const MatrixFlat<float>& A, const MatrixFlat<float>& B, MatrixFlat<float>& C, int64_t& time, int tileSize);

void appendCSVRow(const std::vector<std::string>& rowData, bool newline = false);


void mmm_multiT(const MatrixFlat<float>& A, const MatrixFlat<float>& B, MatrixFlat<float>& C, int64_t& time, int tileSize);
void mmm_multiT(const MatrixFlat<double>& A, const MatrixFlat<double>& B, MatrixFlat<double>& C, int64_t& time, int tileSize);

void mmm_gmultiT(const MatrixFlat<float>& A, const MatrixFlat<float>& B, MatrixFlat<float>& C, int64_t& time, int tileSize, int numThreads = 8);
void mmm_gmultiT(const MatrixFlat<double>& A, const MatrixFlat<double>& B, MatrixFlat<double>& C, int64_t& time, int tileSize, int numThreads = 8);


#endif