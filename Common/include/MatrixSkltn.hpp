#include<vector>
#include<random>
#include<iostream>

#ifndef MATRIXSKTLN_HPP
#define MATRIXSKTLN_HPP


    template<typename T>
    class MatrixSkltn{
    public:
        MatrixSkltn() : n_rows(0), n_cols(0), n_nzrs(0) {};

        //! Added new constructor
        MatrixSkltn (size_t rows,size_t cols, size_t nnzrs):
                n_rows(rows),
                n_cols(cols),
                n_nzrs(nnzrs)
        {};

        size_t nrows() const {return n_rows;}
        size_t ncols() const {return n_cols;}
        size_t nnzrs() const {return n_nzrs;}

    // Print da modificare

        void print(std::ostream& os = std::cout) const {
        os << "nrows: " << n_rows << " | ncols:" << n_cols << " | nnz: " << n_nzrs << std::endl;
        _print(os);
    };
    // Fine Print 

        virtual const T& operator()(size_t i, size_t j) const = 0;
        virtual T& operator()(size_t i, size_t j) = 0;

        virtual ~MatrixSkltn() = default;
    protected:
        size_t n_rows, n_cols, n_nzrs;
        virtual void _print(std::ostream& os) const = 0;

        void generate_random_vector(T a, T b, std::vector<T>& vct);
        void generate_random_vector(T a, T b, std::vector<T>& vct,  int seed);
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
