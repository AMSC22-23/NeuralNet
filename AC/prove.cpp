#include <iostream>
#include<vector>


int main(){

    std::vector<std::vector<int>> test(10,std::vector<int>(5,2));
    for (int i = 0;i<10;i++){
        for (int j=0;j<5;j++){
            std::cout << test[i][j] << " " ;
        }
        std::cout <<std::endl;
    }


    return 0;
}