#include<vector>
#include <cblas.h>

template<typename T> 
class Matrix {

    private:

        std::vector<T> m_data;
        std::size_t rows = 0, cols = 0; 

    public:
        Matrix(std::size_t rows, std::size_t cols, std::vector<T> & data):
            rows(rows),
            cols(cols),
            m_data(data)
            {}; 
    
        T* get_ptr(){return m_data.data(); }
        std::size_t get_rows() const {return rows; }
        std::size_t get_cols() const {return cols; }
        
};




template<typename T>
Matrix<T> mmm_blas(Matrix<T>& A, Matrix<T>& B){

    std::size_t m, n, k; 

    m = A.get_rows(); 
    n = B.get_cols(); 
    k = A.get_cols(); 



    std::vector<T> C(m*n);


    cblas_dgemm(
                CblasRowMajor,      // Specifies row-major (C) or column-major (Fortran) data ordering.
                CblasNoTrans,       // Specifies whether to transpose matrix A.
                CblasNoTrans,       // Specifies whether to transpose matrix B.
                m,             // Number of rows in matrices A and C.
                n,             // Number of columns in matrices B and C.
                k,             // Number of columns in matrix A; number of rows in matrix B.
                1.0,                // Scaling factor for the product of matrices A and B.
                A.get_ptr(),       // UnsafePointer<Double>! to Matrix A.
                k,                  
                B.get_ptr(),     // Matrix B. 
                n,             // The size of the first dimension of matrix B; if you are passing a matrix B[m][n], the value should be m.
                0.0,                // Scaling factor for matrix C.
                C.data(),     // Matrix C.
                n             // The size of the first dimension of matrix C; if you are passing a matrix C[m][n], the value should be m.
                );


     // Stampa la matrice risultato
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << C[i * n + j] << " ";
        }
        std::cout << std::endl;
    }

    return {m, n, C}; 
}; 