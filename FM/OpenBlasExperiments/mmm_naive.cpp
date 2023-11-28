#include "Matrix.hpp"
#include "Profiler.hpp"




int main(){


Matrix<float> A(512, 512); 
Matrix<float> B(512, 512); 

auto dt_blas = Profiler::mmm_blas(A, B); 
auto dt_naive = Profiler::mmm_naive(A, B); 

std::cout<<"Time blas = "<<dt_blas<<"\nTime naive = "<<dt_naive<<std::endl; 

return 0; 
}