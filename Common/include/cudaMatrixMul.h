#ifndef CUDAMMUL_HPP
#define CUDAMMUL_HPP

//*********************************************************************************************************************

// Here you can find the declaration of the functions used to perform matrix multiplication on the GPU,
// the body is defined in /src/cudaMatrixMul.cpp

//*********************************************************************************************************************
void cudaFunctionF(float *a, float *b, float *c, int m, int n, int nb, int block_size);
void cudaFunctionD(double *a, double *b, double *c, int m, int n, int nb, int block_size);
void cudaTileFunctionF(float *a, float *b, float *c, int m, int n, int nb, int block_size);
void cudaTileFunctionD(double *a, double *b, double *c, int m, int n, int nb, int block_size);
void cudaFunctionFOptimized(float *a, float *b, float *c, int m, int n, int nb, int block_size);
void cudaFunctionDOptimized(double *a, double *b, double *c, int m, int n, int nb, int block_size);



#endif