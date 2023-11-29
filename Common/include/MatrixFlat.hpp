#include "MatrixSkltn.hpp"


#ifndef MATRIXFLAT_HPP
#define MATRIXFLAT_HPP



template<typename T>
class MatrixFlat : public MatrixSkltn<T>{
//! This class represent a Matrix, its element are stored contiguously in MM thanks to the use of a single std::vector.
//! This implementation is also compatible with openblas library in which matrix data are stored in a C array.
//! This class do not support nnz count.
    private:

        std::vector<T> m_data;

        bool check_indexes(size_t i, size_t j) const;
    public:


        //! Initializes a matrix with data given as input
        MatrixFlat(size_t rows, size_t cols, const std::vector<T> & data):
            MatrixSkltn<T>(rows, cols, 0),
            m_data(data)
            {};

        //! Initializes a matrix of zeros
        MatrixFlat(std::size_t rows, std::size_t cols):
                MatrixFlat(rows, cols, std::vector<T>(rows*cols))
                    {};

        //! Initializes a matrix filled of random values with values in the interval (a, b)
        MatrixFlat(std::size_t rows, std::size_t cols, T a, T b):
                MatrixSkltn<T>(rows, cols, 0)
            {
                m_data.resize(rows*cols);
                MatrixSkltn<T>::generate_random_vector(a, b, m_data);
            }

        //! Return an unsafe pointer to the data in the heap, useful for interact with openblas library
        T* get_ptr(){return m_data.data(); }


        const T& operator()(size_t i, size_t j) const override;
        T& operator()(size_t i, size_t j) override;

        void _print(std::ostream& os) const override;



    virtual ~MatrixFlat() = default;

};

template<typename T>
bool MatrixFlat<T>::check_indexes(size_t i, size_t j) const {
    // Since using size_t as index time and size_t > 0, it obvius that i, j > 0.
    return i < MatrixSkltn<T>::n_rows &&
            j < MatrixSkltn<T>::n_cols;
}

template<typename T>
void MatrixFlat<T>::_print(std::ostream &os) const {
    std::cout<<"Dim: "<<MatrixSkltn<T>::n_rows<<"X"<<MatrixSkltn<T>::n_cols<<std::endl;
    for (size_t i = 0; i < MatrixSkltn<T>::n_rows; i++) {
        for (size_t j = 0; j < MatrixSkltn<T>::n_cols; j++)
            std::cout << (*this).operator()(i, j) << " ";
        std::cout<<std::endl;
    }
}

template<typename T>
T &MatrixFlat<T>::operator()(size_t i, size_t j) {
    if(check_indexes(i, j) == 1)
        return  m_data[MatrixSkltn<T>::n_rows * i + j];
    std::cerr<<"Error in operator(): indexes "<< i<<", "<< j<< " are not correct.\n"
              <<"Stopping execution. "<<std::endl;
    std::exit(-1);

}

template<typename T>
const T &MatrixFlat<T>::operator()(size_t i, size_t j) const {
    if(check_indexes(i, j) == 1)
        return  m_data[MatrixSkltn<T>::n_rows * i + j];
    std::cerr<<"Error in operator(): indexes "<< i<<", "<< j<< " are not correct.\n"
             <<"Stopping execution. "<<std::endl;
    std::exit(-1);
}


#endif