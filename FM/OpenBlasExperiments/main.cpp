#include <cblas.h>
#include <iostream>

// source /u/sw/etc/bash.bashrc 
// module load gcc-glibc/11.2.0
// module load openblas

/*
 This module provides an optimized implementation of the BLAS and LAPACK
      libraries, it is both multithreaded and optimized for multiple platforms.
      For using OpenBLAS library you should use the linker flags:
        -L$mkOpenblasLib -lopenblas
      By default the library run in single-thread mode, but you can set the number of
        -L$mkOpenblasLib -lopenblas
      libraries, it is both multithreaded and optimized for multiple platforms.
      For using OpenBLAS library you should use the linker flags:
        -L$mkOpenblasLib -lopenblas
      By default the library run in single-thread mode, but you can set the number of
      threads for the multi-thread mode using the proper environment variable:
        export OPENBLAS_NUM_THREADS=4
      or using the proper function to control the number of threads at runtime
        void openblas_set_num_threads(int num_threads);
      Loading this module two utility commands are provided:
        openblas_corename - Return the name of the current architecture
        openblas_numcores - Return the number of physical cores of cpu
      These command can be used to set the optimal value of the OPENBLAS_NUM_THREADS
      variable for your architecture:
        export OPENBLAS_NUM_THREADS=$(openblas_numcores)
*/

int main() {
    // Defining Matrix dimensions
    int rows_A = 3, cols_A = 3, rows_B = 3, cols_B = 3;

    // Inizializing Matrices
    double A[rows_A * cols_A] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    double B[rows_B * cols_B] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    // Inizializing Matrix Result
    double C[rows_A * cols_B];
    
    /* We will use cblas_dgemm
    
        for doc go to:
            https://developer.apple.com/documentation/accelerate/1513282-cblas_dgemm

    */

    // Esecuzione della moltiplicazione delle matrici con OpenBLAS
    cblas_dgemm(
                CblasRowMajor,      // Specifies row-major (C) or column-major (Fortran) data ordering.
                CblasNoTrans,       // Specifies whether to transpose matrix A.
                CblasNoTrans,       // Specifies whether to transpose matrix B.
                rows_A,             // Number of rows in matrices A and C.
                cols_B,             // Number of columns in matrices B and C.
                cols_A,             // Number of columns in matrix A; number of rows in matrix B.
                1.0,                // Scaling factor for the product of matrices A and B.
                A,                  // Matrix A.
                cols_A,             // The size of the first dimension of matrix A; if you are passing a matrix A[m][n], the value should be m.
                B,                  // Matrix B. 
                cols_B,             // The size of the first dimension of matrix B; if you are passing a matrix B[m][n], the value should be m.
                0.0,                // Scaling factor for matrix C.
                C,                  // Matrix C.
                cols_B              // The size of the first dimension of matrix C; if you are passing a matrix C[m][n], the value should be m.
                );

    // Stampare la matrice risultato
    for (int i = 0; i < rows_A; ++i) {
        for (int j = 0; j < cols_B; ++j) {
            std::cout << C[i * cols_B + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
