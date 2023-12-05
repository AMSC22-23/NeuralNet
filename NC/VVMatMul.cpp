#include "VVMatMul.hpp"
#include "MatrixVect2.hpp"
#include <chrono>


// Implementation of naive matrix multiplication
template<typename T>
MatrixVect<T> VVMatMulNaive(const MatrixVect<T>& matrix1, const MatrixVect<T>& matrix2) {
    size_t rows1 = matrix1.nrows();
    size_t cols1 = matrix1.ncols();
    size_t cols2 = matrix2.ncols();

    // Check if matrix dimensions are compatible for multiplication
    if (cols1 != matrix2.nrows()) {
        std::cerr << "Error: Incompatible matrix dimensions for multiplication." << std::endl;
        std::exit(-1);
    }

    // Create a new matrix for the result
    MatrixVect<T> result(rows1, cols2);

    const auto t0 = std::chrono::high_resolution_clock::now();

    // Perform matrix multiplication in a naive manner
    for (size_t row = 0; row < rows1; ++row) {
        for (size_t col = 0; col < cols2; ++col) {
            for (size_t r = 0; r < cols1; ++r) {
                result(row, col) += matrix1(row, r) * matrix2(r, col);
            }
        }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    auto TimeTaken = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Naive multiplication time: " << TimeTaken << " milliseconds\n" << std::endl;

    return result;
}

// Implementation of naive matrix multiplication with accumulation in the register
template<typename T>
MatrixVect<T> VVMatMulNaive2(const MatrixVect<T>& matrix1, const MatrixVect<T>& matrix2) {
    size_t rows1 = matrix1.nrows();
    size_t cols1 = matrix1.ncols();
    size_t cols2 = matrix2.ncols();

    // Check if matrix dimensions are compatible for multiplication
    if (cols1 != matrix2.nrows()) {
        std::cerr << "Error: Incompatible matrix dimensions for multiplication." << std::endl;
        std::exit(-1);
    }

    // Create a new matrix for the result
    MatrixVect<T> result(rows1, cols2);

    const auto t0 = std::chrono::high_resolution_clock::now();

    // Perform matrix multiplication with accumulation in the register
    for (size_t row = 0; row < cols2; ++row) {
        for (size_t col = 0; col < rows1; ++col) {
            T acc = 0.0;
            for (size_t inner = 0; inner < cols1; ++inner) {
                acc += matrix1(row, inner) * matrix2(inner, col);
            }
            result(row, col) = acc;
        }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    auto TimeTaken = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Naive2 multiplication time: " << TimeTaken << " milliseconds\n" << std::endl;
    //result.print();

    return result;
}

// Implementation of matrix multiplication with cache-friendly loop order
template<typename T>
MatrixVect<T> VVMatMulCacheFriendly(const MatrixVect<T>& matrix1, const MatrixVect<T>& matrix2) {
    size_t rows1 = matrix1.nrows();
    size_t cols1 = matrix1.ncols();
    size_t cols2 = matrix2.ncols();

    // Check if matrix dimensions are compatible for multiplication
    if (cols1 != matrix2.nrows()) {
        std::cerr << "Error: Incompatible matrix dimensions for multiplication." << std::endl;
        std::exit(-1);
    }

    // Create a new matrix for the result
    MatrixVect<T> result(rows1, cols2);

    const auto t0 = std::chrono::high_resolution_clock::now();

    // Perform matrix multiplication with cache-friendly loop order
    for (size_t row = 0; row < rows1; ++row) {
        for (size_t inner = 0; inner < cols1; ++inner) {
            for (size_t col = 0; col < cols2; ++col) {
                result(row, col) += matrix1(row, inner) * matrix2(inner, col);
            }
        }
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    auto TimeTaken = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "Cache friendly multiplication time: " << TimeTaken << " milliseconds\n" << std::endl;
    //result.print();

    return result;
}

// Explicit instantiation declarations for double and float types
template MatrixVect<double> VVMatMulNaive(const MatrixVect<double>& matrix1, const MatrixVect<double>& matrix2);
template MatrixVect<float> VVMatMulNaive(const MatrixVect<float>& matrix1, const MatrixVect<float>& matrix2);

template MatrixVect<double> VVMatMulNaive2(const MatrixVect<double>& matrix1, const MatrixVect<double>& matrix2);
template MatrixVect<float> VVMatMulNaive2(const MatrixVect<float>& matrix1, const MatrixVect<float>& matrix2);

template MatrixVect<double> VVMatMulCacheFriendly(const MatrixVect<double>& matrix1, const MatrixVect<double>& matrix2);
template MatrixVect<float> VVMatMulCacheFriendly(const MatrixVect<float>& matrix1, const MatrixVect<float>& matrix2);

int main() {
    
    std::cout << "please choose matrices dimensions" << std::endl;

    int dim;
    std::cin >> dim;
    
    std::cout << std::endl;
    std::cout << "matrices dimensions are " << dim << " rows and " << dim << " columns" << std::endl;
    std::cout << std::endl;
    
    // Build matrix1 and matrix2 with random values between -10 and 10
    MatrixVect<double> m1(dim, dim, -10.0, 10.0);
    MatrixVect<double> m2(dim, dim, -10.0, 10.0);

    // Call the naive matrix multiplication
    auto result_naive = VVMatMulNaive(m1, m2);

    // Call the naive matrix multiplication with accumulation in the register
    auto result_naive2 = VVMatMulNaive2(m1, m2);

    // Call the matrix multiplication with cache-friendly loop order
    auto result_cache_friendly = VVMatMulCacheFriendly(m1, m2);

    return 0;
}