#include<vector>
#include<random>
#include<iostream>

template < typename T>
class Matrix {

    private: 

        std::size_t rows = 0, cols = 0; 
        bool row_major = true; //if true storing is row major, otherwise is col major
        
        
        
        
        std::vector<T> Vdata;

        std::size_t sub2ind(const std::size_t& i,const std::size_t& j) const; 

        bool valid_indexes(const std::size_t&  i, const std::size_t& j) const; 

    public: 

        // Constructors
        Matrix() = default; 
        
        Matrix(const std::size_t& rows, const std::size_t& cols, const bool row_major = true) : 
                        rows(rows), 
                        cols(cols), 
                        row_major(row_major)
                        {   
                            Vdata.resize(rows*cols); 
                         
                        }
    


        // Getter 
        std::size_t get_rows() const { return rows; } 
        std::size_t get_cols() const { return cols; }
        bool get_storage_type() const { return row_major; } 

        // Setter
        void invert_storagetype(){ row_major = !row_major; };


        void random_fill(const T& a = -1., const T& b = 1.); 
        void random_fill(const T& a, const T& b, const int& seed); 

        

        //Operators
        T & operator()(const std::size_t& row, const std::size_t& col) const; 
        T & operator()(const std::size_t& row, const std::size_t& col); 
         
        //Extra
        void print() const ;   
    
}; 


template<typename T> 
bool Matrix<T>::valid_indexes(const std::size_t& i,const std::size_t& j) const{

    return i >= 0 && j>=0 && i < rows && j < cols; 

}

template<typename T> 
std::size_t Matrix<T>::sub2ind(const std::size_t& i,const std::size_t& j) const
{
    // this functions trasforms idexes of matrix to the corresponding index in the vector
    // the oprtation that needs to be performed depends on the type of storage in use
    if(!valid_indexes(i, j))
        {
            std::cerr<<"Wrong indexes! "<<std::endl; 
            std::exit(-1); 
        }

    if (row_major)
        return i*rows + j; 

    return j*cols + i; 
}

template<typename T>
T& Matrix<T>::operator()(const std::size_t& row, const std::size_t& col) const{
    
    std::size_t vect_index = sub2ind(row, col); 
    
    return Vdata[vect_index]; 
}; 


template<typename T>
T& Matrix<T>::operator()(const std::size_t& i, const std::size_t& j){
    
    std::size_t vect_index = sub2ind(i, j); 
    
    return Vdata[vect_index]; 
}; 


template<typename T> 
void Matrix<T>::random_fill( const T& a, const T& b, const int& seed){
    
    std::mt19937 gen(seed); 
    
    std::uniform_real_distribution<T> dist(a, b);
    for(std::size_t i = 0; i<rows*cols; i++)
        Vdata[i] = dist(gen); 
    
}; 



template<typename T> 
void Matrix<T>::random_fill(const T& a , const T& b ){
    
    std::random_device rd;
    random_fill(rd(), a, b); 
}; 


template<typename T> 
void Matrix<T>::print() const {

    std::cout<<"Matrix has dimensions: "<<rows<<"X"<<cols<<std::endl; 
    for(std::size_t i = 0; i<rows; i++)
        {

        for(std::size_t j = 0; j<cols; j++)
            std::cout<<Vdata[sub2ind(i, j)]<<" "; 

        std::cout<<std::endl; 
        
        }
}; 


