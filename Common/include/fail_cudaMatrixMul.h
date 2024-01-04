#ifndef CUDAMMUL_HPP
#define CUDAMMUL_HPP


void cudaFunctionF(float *a, float *b, float *c, int m, int n, int nb, int block_size);
void cudaFunctionD(double *a, double *b, double *c, int m, int n, int nb, int block_size);
void cudaTileFunctionF(float *a, float *b, float *c, int m, int n, int nb, int block_size);
void cudaTileFunctionD(double *a, double *b, double *c, int m, int n, int nb, int block_size);

/**void cudaFunction(float *a, float *b, float *c, int m, int n, int nb, int block_size){
    cudaFunctionF(a, b, c, m, n, nb, block_size);
}
void cudaFunction(double *a, double *b, double *c, int m, int n, int nb, int block_size){
    cudaFunctionD(a, b, c, m, n, nb, block_size);
}**/


#endif