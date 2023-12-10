#include <fstream>
#include <iostream>
#include <sstream>
#include "utilities.hpp"
#include "profiler.hpp"



void Profiler::cachegrindReader() const {

    std::ifstream file(cachegrind_log_filename);

    std::string line;
    std::string result;

    while (getline(file, line)) {
        std::size_t pos = line.find("D1  misses");
        if (pos != std::string::npos) {
            // Here we have found the line we are interested in
            std::size_t start_pos = line.find(':');
            std::size_t end_pos = line.find('(');
            result = line.substr(start_pos + 1, end_pos - start_pos - 1 );

        }
    }
    result = format4csv(result);

    std::cout << result << std::endl;

    append_misses(result);

}

std::string Profiler::format4csv(const std::string& input_string) {
    /*
     *  This function removes extra spaces and commas in order to prepare data to be insterted into csv
     */
    std::string result;

    for (const char c: input_string)
        if(c != ' ' & c != ',')
            result.push_back(c);


    return result;
}

void Profiler::append_misses(const std::string& misses) const {

    std::ofstream file;

    file.open(csv_filename, std::ios_base::app); // opens file in append mode

    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << csv_filename << std::endl;
        return;
    }


    file << "," << misses<<std::endl;

    file.close();

}

void Profiler::profile_one(const std::string& algorithm, const std::string& program_arguments, const std::string& compiler_flags) const  {

    std::string program_filename = algorithm + ".cpp";

    std::cout << "Compiling " + program_filename << std::endl;
    std::cout << "Compiler optimization: " << compiler_flags << std::endl;

    std::string compiling_command = "g++ " + program_filename + " ../../src/mmm.cpp" + compiler_flags + " -o " + algorithm;
    system(compiling_command.data());

    std::cout << "Profiling time complexity" << std::endl;
    std::string run_command = "./" + algorithm + program_arguments + " 0"; // we have to add 0 here because we have to specify to the program that we are executing it not using cachegrind

    system(run_command.data());

    std::cout << "------------------------------------------------------" << std::endl;


    std::string valgrind_call =
            "valgrind --tool=cachegrind --cachegrind-out-file=cachegrindTEMP.txt --log-file=TEMP.txt ./" +
            algorithm + program_arguments + " 1";

    std::cout << "Profiling misses" << std::endl;
    system(valgrind_call.data());

    cachegrindReader();


    std::cout<< "============================================================================================="<< std::endl;

}

void Profiler::profile() const {

    std::ifstream file(profile_list_filename);

    std::string line;

    std::getline(file, line);   // processing separately the first line since it is a header

    while (std::getline(file, line)){

            std::string algorithm, program_arguments, compiler_flags;
            std::istringstream buf(line);

            std::getline(buf, algorithm, ',');
            std::getline(buf, program_arguments, ',');
            std::getline(buf, compiler_flags, ',');

            profile_one(algorithm, program_arguments, compiler_flags);
    }

}







