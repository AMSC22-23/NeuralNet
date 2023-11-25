#include <vector>
#include <cblas.h>
#include <chrono>
#include <random>
#include <iostream>

#ifndef MATRIX_HPP
#define MATRIX_HPP

    template<typename T> 
    class Matrix {

        private:

            std::vector<T> m_data;
            std::size_t rows = 0, cols = 0; 

        public:


            Matrix(std::size_t rows, std::size_t cols, const std::vector<T> & data):
                rows(rows),
                cols(cols),
                m_data(data)
                {}; 
            
            Matrix(std::size_t rows, std::size_t cols):  
                        Matrix(rows, cols, std::vector<T>(rows*cols))
                        {};


            T* get_ptr(){return m_data.data(); }
            std::size_t get_rows() const {return rows; }
            std::size_t get_cols() const {return cols; }
            const T& operator[](std::size_t i) const;  


            void random_fill(T a, T b); 
            void print() const; 
            
    };


template<typename T>
void Matrix<T>::random_fill(T a, T b){
    
    std::random_device rd;
    std::mt19937 gen(rd()); 

    std::uniform_real_distribution<T> dist(a, b);
    for(std::size_t i = 0; i<rows*cols; i++)
        m_data[i] = dist(gen); 

}; 


template<typename T>
void Matrix<T>::print()const{


    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << m_data[i * rows + j] << " ";
        }
        std::cout << std::endl;
    }


}




template<typename T>
const T& Matrix<T>::operator[](std::size_t i) const{

    return m_data[i]; 

}; 




#endif

