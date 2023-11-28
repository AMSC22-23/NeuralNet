#include "Matrix.hpp"
#include <chrono>
#include <string>


class Profiler{

    
    std::string outputfile = "bello.csv"; 
    std::vector<std::size_t> dimensions; 
    std::vector<int64_t> times_float; 
    std::vector<int64_t> times_double; 
    
    
public:
    
    Profiler() = default;
    Profiler(std::vector<std::size_t> dimensions):
            dimensions(dimensions)
            {}; 
    static const int64_t mmm_blas(Matrix<double>& A, Matrix<double>& B); 
    static const int64_t mmm_blas(Matrix<float>& A, Matrix<float>& B); 

    static const int64_t mmm_naive(Matrix<double>& A, Matrix<double>& B); 
    static const int64_t mmm_naive(Matrix<float>& A, Matrix<float>& B);

    // static const int64_t mmm


    void profile(); 

};







