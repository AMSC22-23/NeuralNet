#include <vector>
#include <random>
#include <iostream>

#ifndef MATRIXSKLTN2_HPP
#define MATRIXSKLTN2_HPP

template <typename T>
class MatrixSkltn {
public:
   
   
    MatrixSkltn() : n_rows(0), n_cols(0) {};

    MatrixSkltn(size_t rows, size_t cols)
        : n_rows(rows), n_cols(cols) {};

    size_t nrows() const { return n_rows; }
    size_t ncols() const { return n_cols; }
    
    
    virtual const T& operator()(size_t i, size_t j) const = 0;
    virtual T& operator()(size_t i, size_t j) = 0;
    

protected:
    
    size_t n_rows, n_cols;

    virtual void print(std::ostream& os) const = 0;

    // Define a default implementation for generate_random_vector
    void generate_random_vector(T a, T b, std::vector<T>& vct, int seed);

    void generate_random_vector(T a, T b, std::vector<T>& vct);

};

template<typename T> 
    void MatrixSkltn<T>::generate_random_vector(T a, T b, std::vector<T>& vct, int seed){

        std::mt19937 gen(seed); 
        std::uniform_real_distribution<T> dist(a, b);

        for(std::size_t i = 0; i<vct.size(); i++)
            vct[i] = dist(gen); 


    }

    template<typename T> 
    void MatrixSkltn<T>::generate_random_vector(T a, T b, std::vector<T>& vct){

        std::random_device rd;
        generate_random_vector(a, b, vct, rd());

    }

#endif
