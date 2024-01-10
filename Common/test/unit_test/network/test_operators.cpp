#include "functions_utilities.hpp"
#include <vector>
#include <iostream>



int main(){

#ifdef _OPENMP
    std::cout<<"Using openmp"<<std::endl;
#endif

    std::vector<double> a (5, 1);
    std::vector<double> b (5, 2);

    std::vector<double> c = a + b;

    for (const double elem: c)
        std::cout<<elem<< " "<<std::endl;


    return 0;
}