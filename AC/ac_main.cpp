#include <iostream>
#include <cstring>
#include "matrixgen.hpp"

int main(int argc, char **argv){
    
    if (argc < 3) {
        std::cerr << "parametri insufficienti in: " << argv[0] << std::endl;
        return 1;
    }
    if (argc == 4 ){
     if (std::strcmp(argv[3], "f") != 0 && std::strcmp(argv[3], "d") != 0) {
        std::cerr << "parametro non supportato " << argv[3] << std::endl;
        return 1;
     }
    }
    if (argc < 4) {
        std::cout << "Valori della matrice impostati di default su -float- " << std::endl;
    }
    
    std::cout << "numero parametri " << argc << std::endl;
    for(int i;i<argc;i++){
        std::cout << "parametro " << i << "valore: " << argv[i] << std::endl;
    }

     try {
        int m = std::stoi(argv[1]);
        int n = std::stoi(argv[2]);
        if(std::strcmp(argv[3], "f") == 0){
            auto matrixF = matrixgen<float>(m, n);
            for (const auto& riga : matrixF) {
                for (auto valore : riga) {
                    std::cout << valore << " ";
                }
             std::cout << std::endl;
            }
        }
        if(std::strcmp(argv[3], "d") == 0){
            auto matrixD = matrixgen<float>(m, n);
            for (const auto& riga : matrixF) {
                for (auto valore : riga) {
                    std::cout << valore << " ";
                }
                std::cout << std::endl;
            }
        }
        std::cout << "matrice definita m x n: " << m << " x " << n << std::endl;
        
    } catch (const std::invalid_argument& e) {
        std::cerr << "Errore: Conversione fallita " << e.what() << std::endl;
        return 1;
    } catch (const std::out_of_range& e) {
        std::cerr << "Errore: Numero non consentito " << e.what() << std::endl;
        return 1;
    }

    
    return 0;
}