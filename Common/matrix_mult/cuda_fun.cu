#include "cuda_launcher.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#define tile 32


__global__ void gpu_matrix_mult(float *a,float *b, float *c, int m, int n, int nb)
{
    //if(threadIdx.y==0)
      //printf("inizio con valore %f %f %f\n",a[0], a[1], a[2]);
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

__global__ void gpu_matrix_mult_tile(float *a,float *b, float *c, int m, int n, int nb)
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

void test(){
    printf("vediamo se funzionano anche le funzioni++++++++++++");
}

void cudaFunction(float *a, float *b, float *c, int m, int n, int nb){
    //std::cout << "vediamo se funziona " << a[0] << " " << b[0] <<std::endl;
    printf("passati valori %f %f\n",a[0],a[1]);
    float *ac,*bc,*cc;

    cudaMallocManaged((void **) &ac, sizeof(float)*m*n);
    cudaMallocManaged((void **) &bc, sizeof(float)*n*nb);
    cudaMallocManaged((void **) &cc, sizeof(float)*m*nb);

    cudaMemcpy(ac, a, sizeof(float) * m *n, cudaMemcpyHostToDevice);
    cudaMemcpy(bc, b, sizeof(float) * n *nb, cudaMemcpyHostToDevice);
    //cudaMemcpy(cc, c, sizeof(float) * m *nb, cudaMemcpyHostToDevice);

    printf("in teoria funziona...\n");
    //test();
    //std::cout << "salvate in teoria..." << std::endl;

    float  naive_gpu_elapsed_time_ms;

    // some events to count the execution time
    //clock_t st, end;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int MATRIX_SIZE_X = m;
    int MATRIX_SIZE_Y = nb;

    for(int block_size= 4; block_size <= 32; block_size *= 2){
        unsigned int grid_rows = (MATRIX_SIZE_X + block_size - 1) / block_size;
        unsigned int grid_cols = (MATRIX_SIZE_Y + block_size - 1) / block_size;
        dim3 dimGrid(grid_cols, grid_rows);
        dim3 dimBlock(block_size, block_size);

        cudaEventRecord(start, 0);
        gpu_matrix_mult<<<dimGrid, dimBlock>>>(ac, bc, cc, m, n, nb);
        cudaThreadSynchronize();

        // time counting terminate

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        // compute time elapsed on GPU computing
        cudaEventElapsedTime(&naive_gpu_elapsed_time_ms, start, stop);
        printf("Time elapsed on naive GPU matrix multiplication of %dx%d . %dx%d (%d): %f ms.\n\n", m, n, n, nb, block_size, naive_gpu_elapsed_time_ms);

        
        cudaEventRecord(start, 0);
        gpu_matrix_mult_tile<<<dimGrid, dimBlock>>>(ac, bc, cc, m, n, nb);
        cudaThreadSynchronize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&naive_gpu_elapsed_time_ms, start, stop);
        printf("Time elapsed on tiled (%d) GPU matrix multiplication of %dx%d . %dx%d (%d): %f ms.\n\n",tile, m, n, n, nb, block_size, naive_gpu_elapsed_time_ms);

        

    }


    cudaMemcpy(c, cc, sizeof(float) * m*nb, cudaMemcpyDeviceToHost);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

}