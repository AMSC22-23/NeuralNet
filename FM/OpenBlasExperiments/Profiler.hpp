#include "Matrix.hpp"
#include <chrono>
#include <string>


class Profiler{

    
    std::string outputfile = "bello.csv"; 
    std::vector<std::size_t> dimensions; 
    std::vector<int64_t> times_float; 
    std::vector<int64_t> times_double; 
    
    inline void write_result(const std::string& algorithmID, std::size_t n, const std::string& datatype, int64_t dt) const; 
public:
    
    Profiler() = default;
    Profiler(std::vector<std::size_t> dimensions):
            dimensions(dimensions)
            {}; 
    const int64_t mmm_blas(Matrix<double>& A, Matrix<double>& B); 
    const int64_t mmm_blas(Matrix<float>& A, Matrix<float>& B); 

    const int64_t mmm_naive(Matrix<double>& A, Matrix<double>& B); 
    void profile(); 

};







