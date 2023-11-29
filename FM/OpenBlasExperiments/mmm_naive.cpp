#include "Matrix.hpp"
#include "Profiler.hpp"


/*
Without any optimizations:
Time blas = 8
Time naive = 1946
------------------------------------------
Loop unrolling all loops:      -funroll-all-loops
Time blas = 4
Time naive = 1954
--------------------------------------------------------
Loop unrolling loops whos iteration can be found at compiletime:      -funroll-loops
Time blas = 4
Time naive = 1856

Time blas = 4
Time naive = 1856

------------------------------------------------------------
-O3 
Time blas = 4
Time naive = 235
------------------------------------------------------------
-Ofast
Time blas = 4
Time naive = 235

*/



int main(){


Matrix<float> A(3000, 3000); 
Matrix<float> B(3000, 3000); 

auto dt_blas = Profiler::mmm_blas(A, B); 
// auto dt_naive = Profiler::mmm_naive(A, B); 

std::cout<<"Time blas = "<<dt_blas<<"\nTime naive = "<<std::endl; 

return 0; 
}