#include <algorithm>
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

        inline static bool is_zero(T elem, T tolerance);

        void compute_nzrs(T tolerance = 1e-10);


    public:


        //! Initializes a matrix with data given as input
        MatrixFlat(size_t rows, size_t cols, const std::vector<T> & data):
            MatrixSkltn<T>(rows, cols, 0),
            m_data(data)
            {
                compute_nzrs();

            };

        //! Initializes a matrix of zeros
        MatrixFlat(std::size_t rows, std::size_t cols):
            MatrixSkltn<T>(rows, cols, 0),
            m_data(std::vector<T>(rows*cols))
            {};

        //! Initializes a matrix filled of random values with values in the interval (a, b)
        MatrixFlat(std::size_t rows, std::size_t cols, T a, T b):
                MatrixSkltn<T>(rows, cols, 0)
            {
                m_data.resize(rows*cols);
                MatrixSkltn<T>::generate_random_vector(a, b, m_data);
                compute_nzrs();
            }

        //! Return an unsafe pointer to the data in the heap, useful for interact with openblas library
        T* get_ptr(){return m_data.data(); }


        const T& operator()(size_t i, size_t j) const override;
        T& operator()(size_t i, size_t j) override;

        inline const T& operator[](size_t index) const;
        inline T& operator[](size_t index);
        void _print(std::ostream& os) const override;

        //aggiunto da ale, ritorno il vettore dati
        std::vector<T> getMdata(){return m_data; }


        //Aggiunto da fil
        size_t nnzrs() override;

        //aggiunto da fil, necessario per verificare correttezza mmm
        MatrixFlat<T> operator-(const MatrixFlat<T>& B) const;

    virtual ~MatrixFlat() = default;

};

template<typename T>
MatrixFlat<T> MatrixFlat<T>::operator-(const MatrixFlat<T> &B) const {
    if(this->nrows() != B.nrows() || this->ncols() != B.ncols())
        {
        std::cerr<<"Error: dimension of the two matrices are wrong: cannot compute difference. Stopping execution. "<<std::endl;
        std::exit(-1);
        }
    std::vector<T> diff( B.nrows() * B.ncols());
    for(size_t i = 0; i < B.nrows() * B.ncols(); i++ )
        diff[i] = (*this)[i] - B[i];

    return MatrixFlat<T>( B.nrows(),  B.ncols(), diff);
}

template<typename T>
size_t MatrixFlat<T>::nnzrs()  {
    compute_nzrs();
    return MatrixSkltn<T>::nnzrs();
}

template<typename T>
bool MatrixFlat<T>::is_zero(T elem, T tolerance ) {
    return std::abs(elem) <= tolerance;
}


template<typename T>
void MatrixFlat<T>::compute_nzrs(T tolerance) {
    unsigned int count = 0;
    for (const T& elem : m_data)
    {
        count += !is_zero(elem, tolerance);
    }
   MatrixSkltn<T>::n_nzrs = count;
}

template<typename T>
T &MatrixFlat<T>::operator[](size_t index) {
    return m_data[index];
}

template<typename T>
const T &MatrixFlat<T>::operator[](size_t index) const {
    return m_data[index];
}

template<typename T>
bool MatrixFlat<T>::check_indexes(size_t i, size_t j) const {
    // Since using size_t as index time and size_t > 0, it obvius that i, j > 0.
    return i < MatrixSkltn<T>::n_rows &&
            j < MatrixSkltn<T>::n_cols;
}

template<typename T>
void MatrixFlat<T>::_print(std::ostream &os) const {

    for (size_t i = 0; i < MatrixSkltn<T>::n_rows; i++) {
        for (size_t j = 0; j < MatrixSkltn<T>::n_cols; j++)
            std::cout << (*this).operator()(i, j) << " ";
        std::cout<<std::endl;
    }
}

template<typename T>
T &MatrixFlat<T>::operator()(size_t i, size_t j) {
    //@note: check_indexes is expensive, consider coding also a version that does not have this overhead
    if(check_indexes(i, j) == 1)
        return  m_data[MatrixSkltn<T>::n_cols * i + j];
    std::cerr<<"Error in operator(): indexes "<< i<<", "<< j<< " are not correct.\n"
              <<"Stopping execution. "<<std::endl;
    std::exit(-1);

}

template<typename T>
const T &MatrixFlat<T>::operator()(size_t i, size_t j) const {
    if(check_indexes(i, j) == 1)
        return  m_data[MatrixSkltn<T>::n_cols * i + j];
    std::cerr<<"Error in operator(): indexes "<< i<<", "<< j<< " are not correct.\n"
             <<"Stopping execution. "<<std::endl;
    std::exit(-1);
}


#endif