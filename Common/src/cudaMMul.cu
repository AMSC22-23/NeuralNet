#include "../include/cudaMatrixMul.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#define tile 32


__global__ void gpu_matrix_multF(float *a,float *b, float *c, int m, int n, int nb)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < nb && row < m)
    {
        int sum = 0;
        for(int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * nb + col];
        }
        c[row * nb + col] = sum;
    }
}

__global__ void gpu_matrix_multD(double *a,double *b, double *c, int m, int n, int nb)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if( col < nb && row < m)
    {
        int sum = 0;
        for(int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * nb + col];
        }
        c[row * nb + col] = sum;
    }
}

__global__ void gpu_matrix_mult_tileF(float *a,float *b, float *c, int m, int n, int nb)
{
    __shared__ int ds_M[tile][tile];
    __shared__ int ds_N[tile][tile];


  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int Row = by * blockDim.y + ty;
  int Col = bx * blockDim.x + tx;
  int Pvalue = 0;

  // Loop over the M and N tiles required to compute the P element
  for (int p = 0; p < (n-1) / tile + 1; ++p) {
    // Collaborative loading of M and N tiles into shared memory
    if(Row < m && p * tile+tx < n) {
        ds_M[ty][tx] = a[Row*n + p*tile+tx];
    }
    else
    {
        ds_M[ty][tx] = 0.0;
    }
    if (p*tile+ty < n && Col < nb) {
        ds_N[ty][tx] = b[(p*tile+ty)*nb + Col];
    }
    else
    {
        ds_N[ty][tx] = 0.0;
    }
    __syncthreads();

    if(Row < m && Col < nb) {
        for (int i = 0; i < tile; ++i)
           Pvalue += ds_M[ty][i] * ds_N[i][tx];
    }
    __syncthreads();
  }
  if (Row < m && Col < nb)
    c[Row*nb+Col] = Pvalue;
}

__global__ void gpu_matrix_mult_tileD(double *a,double *b, double *c, int m, int n, int nb)
{
    __shared__ int ds_M[tile][tile];
    __shared__ int ds_N[tile][tile];


  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  int Row = by * blockDim.y + ty;
  int Col = bx * blockDim.x + tx;
  int Pvalue = 0;

  // Loop over the M and N tiles required to compute the P element
  for (int p = 0; p < (n-1) / tile + 1; ++p) {
    // Collaborative loading of M and N tiles into shared memory
    if(Row < m && p * tile+tx < n) {
        ds_M[ty][tx] = a[Row*n + p*tile+tx];
    }
    else
    {
        ds_M[ty][tx] = 0.0;
    }
    if (p*tile+ty < n && Col < nb) {
        ds_N[ty][tx] = b[(p*tile+ty)*nb + Col];
    }
    else
    {
        ds_N[ty][tx] = 0.0;
    }
    __syncthreads();

    if(Row < m && Col < nb) {
        for (int i = 0; i < tile; ++i)
           Pvalue += ds_M[ty][i] * ds_N[i][tx];
    }
    __syncthreads();
  }
  if (Row < m && Col < nb)
    c[Row*nb+Col] = Pvalue;
}





void cudaFunctionF(float *a, float *b, float *c, int m, int n, int nb, int block_size){
    float *ac,*bc,*cc;

    cudaMallocManaged((void **) &ac, sizeof(float)*m*n);
    cudaMallocManaged((void **) &bc, sizeof(float)*n*nb);
    cudaMallocManaged((void **) &cc, sizeof(float)*m*nb);

    cudaMemcpy(ac, a, sizeof(float) * m *n, cudaMemcpyHostToDevice);
    cudaMemcpy(bc, b, sizeof(float) * n *nb, cudaMemcpyHostToDevice);
    //cudaMemcpy(cc, c, sizeof(float) * m *nb, cudaMemcpyHostToDevice);

    int MATRIX_SIZE_X = m;
    int MATRIX_SIZE_Y = nb;
    unsigned int grid_rows = (MATRIX_SIZE_X + block_size - 1) / block_size;
    unsigned int grid_cols = (MATRIX_SIZE_Y + block_size - 1) / block_size;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(block_size, block_size);

    gpu_matrix_multF<<<dimGrid, dimBlock>>>(ac, bc, cc, m, n, nb);
    cudaThreadSynchronize();

    cudaMemcpy(c, cc, sizeof(float) * m*nb, cudaMemcpyDeviceToHost);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

}


void cudaFunctionD(double *a, double *b, double *c, int m, int n, int nb, int block_size){

    double *ac,*bc,*cc;

    cudaMallocManaged((void **) &ac, sizeof(double)*m*n);
    cudaMallocManaged((void **) &bc, sizeof(double)*n*nb);
    cudaMallocManaged((void **) &cc, sizeof(double)*m*nb);

    cudaMemcpy(ac, a, sizeof(double) * m *n, cudaMemcpyHostToDevice);
    cudaMemcpy(bc, b, sizeof(double) * n *nb, cudaMemcpyHostToDevice);
    //cudaMemcpy(cc, c, sizeof(float) * m *nb, cudaMemcpyHostToDevice);

    int MATRIX_SIZE_X = m;
    int MATRIX_SIZE_Y = nb;

    unsigned int grid_rows = (MATRIX_SIZE_X + block_size - 1) / block_size;
    unsigned int grid_cols = (MATRIX_SIZE_Y + block_size - 1) / block_size;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(block_size, block_size);

    gpu_matrix_multD<<<dimGrid, dimBlock>>>(ac, bc, cc, m, n, nb);
    cudaThreadSynchronize();

}


void cudaTileFunctionF(float *a, float *b, float *c, int m, int n, int nb, int block_size){
    
    float *ac,*bc,*cc;

    cudaMallocManaged((void **) &ac, sizeof(float)*m*n);
    cudaMallocManaged((void **) &bc, sizeof(float)*n*nb);
    cudaMallocManaged((void **) &cc, sizeof(float)*m*nb);

    cudaMemcpy(ac, a, sizeof(float) * m *n, cudaMemcpyHostToDevice);
    cudaMemcpy(bc, b, sizeof(float) * n *nb, cudaMemcpyHostToDevice);

    int MATRIX_SIZE_X = m;
    int MATRIX_SIZE_Y = nb;

    unsigned int grid_rows = (MATRIX_SIZE_X + block_size - 1) / block_size;
    unsigned int grid_cols = (MATRIX_SIZE_Y + block_size - 1) / block_size;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(block_size, block_size);

    gpu_matrix_mult_tileF<<<dimGrid, dimBlock>>>(ac, bc, cc, m, n, nb);
        
    cudaMemcpy(c, cc, sizeof(float) * m*nb, cudaMemcpyDeviceToHost);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

}

void cudaTileFunctionD(double *a, double *b, double *c, int m, int n, int nb, int block_size){
    double *ac,*bc,*cc;

    cudaMallocManaged((void **) &ac, sizeof(double)*m*n);
    cudaMallocManaged((void **) &bc, sizeof(double)*n*nb);
    cudaMallocManaged((void **) &cc, sizeof(double)*m*nb);

    cudaMemcpy(ac, a, sizeof(double) * m *n, cudaMemcpyHostToDevice);
    cudaMemcpy(bc, b, sizeof(double) * n *nb, cudaMemcpyHostToDevice);
    //cudaMemcpy(cc, c, sizeof(float) * m *nb, cudaMemcpyHostToDevice);

    int MATRIX_SIZE_X = m;
    int MATRIX_SIZE_Y = nb;

    unsigned int grid_rows = (MATRIX_SIZE_X + block_size - 1) / block_size;
    unsigned int grid_cols = (MATRIX_SIZE_Y + block_size - 1) / block_size;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(block_size, block_size);

    gpu_matrix_mult_tileD<<<dimGrid, dimBlock>>>(ac, bc, cc, m, n, nb);
    cudaMemcpy(c, cc, sizeof(double) * m*nb, cudaMemcpyDeviceToHost);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

}