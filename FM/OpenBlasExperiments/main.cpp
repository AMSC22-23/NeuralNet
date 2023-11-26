#include <cblas.h>
#include <iostream>
#include "Matrix.hpp"
#include<vector>


/*
  this is a simple example of an application of the openblas library. 
  we compute the product of two matrices. 

  To load the module 

       source /u/sw/etc/bash.bashrc && module load gcc-glibc/11.2.0 && module load openblas

  This flags are needed by the linker     
     -L$mkOpenblasLib -lopenblas

  This flags are needed by compiler
      -I ${mkOpenblasInc}


  To compile the program
    g++ main.cpp  -I ${mkOpenblasInc}  -L${mkOpenblasLib} -lopenblas -o program


  OPENBLAS MODULE:

   This module provides an optimized implementation of the BLAS and LAPACK
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



/*
  DGEMM  performs one of the matrix-matrix operations

    C := alpha*op( A )*op( B ) + beta*C,

   where  op( X ) is one of

    op( X ) = X   or   op( X ) = X**T,

  alpha and beta are scalars, and A, B and C are matrices, with op( A )
  an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.

*/








int main() {
    // Defining Matrix dimensions
   


    
    std::vector<double> Avect {5, 0 , 0, 
                      0, 6, 0, 
                      0, 0, 7, 
                      0, 0, 0};


    std::vector<double> Bvect {5, 0 , 0, 0, 
                              0, 6, 0, 0,
                               0, 0, 7, 0};


    Matrix<double> A(4, 3, Avect); 
    Matrix<double> B(3, 4, Bvect); 

    Matrix<double> C = mmm_blas(A, B);

    std::size_t m, n, k; 

    m = 4; 
    k = 3; 
    n = 4; 

    std::vector<double> Cvect (m*n); 
    //double C[m * n];

    /*

    cblas_dgemm(
                CblasRowMajor,      // Specifies row-major (C) or column-major (Fortran) data ordering.
                CblasNoTrans,       // Specifies whether to transpose matrix A.
                CblasNoTrans,       // Specifies whether to transpose matrix B.
                m,             // Number of rows in matrices A and C.
                n,             // Number of columns in matrices B and C.
                k,             // Number of columns in matrix A; number of rows in matrix B.
                1.0,                // Scaling factor for the product of matrices A and B.
                Avect.data(),       // UnsafePointer<Double>! to Matrix A.
                k,                  
                Bvect.data(),     // Matrix B. 
                n,             // The size of the first dimension of matrix B; if you are passing a matrix B[m][n], the value should be m.
                0.0,                // Scaling factor for matrix C.
                Cvect.data(),     // Matrix C.
                n             // The size of the first dimension of matrix C; if you are passing a matrix C[m][n], the value should be m.
                );

    // Stampa la matrice risultato
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << Cvect[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

  */


  


    return 0;
}
