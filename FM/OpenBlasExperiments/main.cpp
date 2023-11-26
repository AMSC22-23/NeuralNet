#include <cblas.h>
#include <iostream>
#include "Matrix.hpp"
#include "Profiler.hpp"
#include<vector>
#include <chrono>

/*
  
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




int main(int argc, char** argv) {

  
/*
  openblas_set_num_threads(4);

  Profiler p ({4}); 


  p.profile(); 

  */


 Matrix<float> A(2, 2); 
 Matrix<float> B(2, 2); 
A.random_fill(-2, 2);
B.random_fill(-2, 2);


A.print();

std::cout<<"-----------------------------------------------------\n"; 

B.print();


  return 0;
}
