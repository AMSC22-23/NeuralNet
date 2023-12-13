#include<iostream>
#include<vector>
#include<string>


template<typename T>
class Input{
    public:
    
    void setInputSet(std::vector<std::vector<T>> train, std::vector<std::vector<T>> validation, std::vector<std::vector<T>> test,
            std::vector<std::vector<T>>& train_input, std::vector<std::vector<T>>& validation_input,
             std::vector<std::vector<T>>& test_input) {   //sono di tentativo:vanno implementate !
        
        train_input = train;
        test_input = test;
        validation_input = validation;
    }

    std::vector<std::vector<T>> getTrain(){return train_input;}
    std::vector<std::vector<T>> getTest(){return test_input;}
    std::vector<std::vector<T>> getValidation(){return validation_input;}


    void setShape(std::vector<std::vector<T>> input, std::vector<int>& output) {
        output[0] = input.size();
        output[1] = input[0].size();
    }

    Input(std::vector<std::vector<T>> train, std::vector<std::vector<T>> validation, std::vector<std::vector<T>> test) 
            {setInputSet(train, validation, test, train_input, validation_input, test_input),
            setShape(train, train_shape),
            setShape(validation, validation_shape),
            setShape(test, test_shape);};

    Input(const Input<T>& input){
        train_input = input.getTrain();
        test_input = input.getTest();
        validation_input = input.getValidation();
        train_shape[0] = train_input.size();
        train_shape[1] = train_input[0].size();
        validation_shape[0] = validation_input.size();
        validation_shape[1] = validation_input[0].size();
        test_shape[0] = test_input.size();
        test_shape[1] = test_input[0].size();
    };
            
    void printShape() const {
        std::cout << "Shape of the different input sets:" << std::endl;
        std::cout << "Train set: " << "[ " << Input<T>::train_shape[0] << " , " << train_shape[1] << " ]" << std::endl;
        std::cout << "Validation set: " << "[ " << Input<T>::validation_shape[0] << " , " << validation_shape[1] << " ]" << std::endl;
        std::cout << "Test set: " << "[ " << Input<T>::test_shape[0] << " , " << test_shape[1] << " ]" << std::endl;

    }

    int getShapeInputData() const {return train_shape[1];}

    std::vector<std::vector<T>> getTrain() const {return train_input;}
    std::vector<std::vector<T>> getTest() const {return test_input;}
    std::vector<std::vector<T>> getValidation() const {return validation_input;}


    private:
    std::vector<std::vector<T>> train_input, validation_input, test_input;
    std::vector<int> train_shape{0,0}, validation_shape{0,0}, test_shape{0,0};
};

template<typename T>
class Output{
    public:
    Output(std::vector<std::vector<T>> train_target, std::vector<std::vector<T>> validadion_target,
         std::vector<std::vector<T>> test_target, std::string sel_actFun)
     {setTarget(train_target,output_target_train);
    setTarget(validadion_target,output_target_validation);
    setTarget(test_target,output_target_test);
    setShape(train_target, train_shape);
    setShape(validadion_target, validation_shape);
    setShape(test_target, test_shape);
    set_Act_Fun(sel_actFun, act_funct);};

    Output(const Output<T>& copy){
        output_target_train = copy.getOutputTrain();
        output_target_test = copy.getOutputTest();
        output_target_validation = copy.getOutputValidation();
        train_shape[0] = output_target_train.size();
        train_shape[1] = output_target_train[0].size();
        validation_shape[0] = output_target_validation.size();
        validation_shape[1] = output_target_validation[0].size();
        test_shape[0] = output_target_test.size();
        test_shape[1] = output_target_test[0].size();
        act_funct = copy.getOutputAct_fun();
        
    };
    


    void setTarget(std::vector<std::vector<T>> target, std::vector<std::vector<T>>& output_target) {
        output_target = target;
    }

    void setShape(std::vector<std::vector<T>> input, std::vector<int>& output) {
        output[0] = input.size();
        output[1] = input[0].size();
    }

    std::vector<std::vector<T>> getOutputTrain() const {return output_target_train;}
    std::vector<std::vector<T>> getOutputTest() const {return output_target_test;}
    std::vector<std::vector<T>> getOutputValidation() const {return output_target_validation;}
    std::string getOutputAct_fun() const {return act_funct;}
   

    void setResult(std::vector<T> new_result, std::vector<T>& output_result) {
        output_result.push_back(new_result);
    }

    void set_Act_Fun(std::string selection, std::string& variable){
        variable = selection;
    }

    void printShape() const {
        std::cout << "Shape of the different output sets:" << std::endl;
        std::cout << "Train output set: " << "[ " << Output<T>::train_shape[0] << " , " << train_shape[1] << " ]" << std::endl;
        std::cout << "Validation output set: " << "[ " << Output<T>::validation_shape[0] << " , " << validation_shape[1] << " ]" << std::endl;
        std::cout << "Test output set: " << "[ " << Output<T>::test_shape[0] << " , " << test_shape[1] << " ]" << std::endl;
        std::cout << "Activation function applied on the output: " << Output<T>::act_funct << std::endl;

    }

    int getShapeOutputData() const {return train_shape[1];}
    std::string getAct_Fun() const {return act_funct;}
    

    private:
    std::vector<std::vector<T>> output_target_train, output_target_validation, output_target_test, output_result;
    std::vector<int> train_shape{0,0}, validation_shape{0,0}, test_shape{0,0};
    std::string act_funct;

};

//template<typename T>
class Layer{
    public:
    Layer(std::string layer_name, int num_of_neurons, std::string layer_act_funct):
        name(layer_name), neurons(num_of_neurons), act_funct(layer_act_funct) {};

    void printLayer() const {
        std::cout << "Layer: " << name << " ,number of neurons: " << neurons << " ,activation function: " << act_funct << std::endl;
    }

    int getNeurons() {return neurons;}
    std::string getActFun() {return act_funct;};
    private:
    std::string act_funct, name;
    bool dense = true;
    int neurons;

};
