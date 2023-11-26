#include "Matrix.hpp"
#include <chrono>



class Profiler{

    

    std::vector<std::size_t> dimensions; 


public:
    
    Profiler() = default;
    Profiler(std::vector<std::size_t> dimensions):
            dimensions(dimensions)
            {}; 
    const int64_t mmm_blas(Matrix<double>& A, Matrix<double>& B); 
    //Matrix<float> mmm_blas(Matrix<float>& A, Matrix<float>& B); 

    void profile(); 

};







