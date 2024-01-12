#include 'model.hpp'


template<typename T> 
void Model<T>::predict(std::vector<T>& input, const int& selection){
    input.push_back(1);
    mul_funct(input, weights[0], z[0],
         1, weights_shape[0][0]+1, weights_shape[0][1], matrix_mul_optimisation);  
    activationFun(z[0], h[0], layers[0].getActFun());
    for(int loop = 0; loop < layers.size(); loop++){
        mul_funct(h[loop], weights[loop+1], z[loop+1],
             1, weights_shape[loop+1][0]+1, weights_shape[loop+1][1], matrix_mul_optimisation);
       if(loop < layers.size()-1){
            activationFun(z[loop+1], h[loop+1], layers[loop+1].getActFun());
        }
    }
    activationFun(z[layers.size()], y, model_output.getOutputAct_fun());
    input.pop_back();
}