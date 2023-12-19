#include <fstream>
#include <iostream>
#include <sstream>
//#include "utilities.hpp"
#include "../include/profiler.hpp"



void Profiler::cachegrindReader() const {

    /*
     * This function is needed to parse the output of valgrind cachegrind.
     * It reads the file cachegrind_log_filename and extracts the number of cache misses.
     *
     */

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
     *  This function removes extra spaces and commas in order to prepare data to be inserted into csv
     */
    std::string result;

    for (const char c: input_string)
        if(c != ' ' & c != ',')
            result.push_back(c);


    return result;
}

void Profiler::append_misses(const std::string& misses) const {

    /*
     * This function appends the number of misses to the csv file
     */

    std::ofstream file;

    file.open(csv_filename, std::ios_base::app); // opens file in append mode

    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << csv_filename << std::endl;
        return;
    }


    file << "," << misses<<std::endl;

    file.close();

}

void Profiler::profile_one(const std::string& algorithm, const std::string& program_arguments, const std::string& compiler_flags, bool profile_misses) const {

    /*
     * This method profiles one line of the profile_list_filename file, which must be passed as input already parsed.
     * Since we want to be able to test different compiler flags we need each time to compile the program.
     * Here we compile the program by calling g++ using the system function : https://cplusplus.com/reference/cstdlib/system/
     *
     * We want to profile time complexity but also the # of cache misses, so we use valgrind cachegrind to do so.
     * Since when profiling misses with valgrind the execution time grows a lot, we need to execute the program twice:
     * the first time we execute it without valgrind, and the second time we execute it with valgrind.
     *
     * The misses are profiled only if profile_misses is true.
     *
     * The time complexity is not measured directly by the profiler class but by the program itself.
     * Since when we profile misses we don't want to write the execution time on the output, then we need to pass
     * an extra argument to the program, which is 0 if we are profiling time complexity and 1 if we are profiling
     * Then based on this argument the program decides whether to write the execution time or not.
     */


    std::string program_filename = algorithm + ".cpp";
    std::string openblas_flags = " -lopenblas ";

    std::cout << "Compiling " + program_filename << std::endl;
    std::cout << "Compiler optimization: " << compiler_flags << std::endl;

    std::string compiling_command =
            "g++ " + program_filename + " ../../src/mmm.cpp ../../src/mmm_blas.cpp" + compiler_flags + " -o " + algorithm + openblas_flags;
    system(compiling_command.data());

    std::cout << "Profiling time complexity" << std::endl;
    std::string run_command = "./" + algorithm + program_arguments +
                              " 0"; // we have to add 0 here because we have to specify to the program that we are executing it not using cachegrind

    system(run_command.data());

    std::cout << "------------------------------------------------------" << std::endl;

    if (profile_misses){
        std::string valgrind_call =
                "valgrind --tool=cachegrind --cachegrind-out-file=cachegrindTEMP.txt --log-file=TEMP.txt ./" +
                algorithm + program_arguments + " 1";

    std::cout << "Profiling misses" << std::endl;
    system(valgrind_call.data());

    cachegrindReader();

    }
    else{
        std::cout << "Misses profiling skipped" << std::endl;
        append_misses("-1");
    }
    std::cout<< "============================================================================================="<< std::endl;

}

void Profiler::profile() const {

    /*
     * This method executes the profiling of the algorithms listed in the file profile_list_filename
     * and writes the results in the output files
     */

    std::ifstream file(profile_list_filename);

    std::string line;

    std::getline(file, line);   // processing separately the first line since it is a header

    while (std::getline(file, line)){

            std::string algorithm, program_arguments, compiler_flags, profile_misses_string;
            std::istringstream buf(line);

            std::getline(buf, algorithm, ',');
            std::getline(buf, program_arguments, ',');
            std::getline(buf, compiler_flags, ',');

            std::getline(buf, profile_misses_string, ',');

            bool profile_misses = (profile_misses_string == "1");


            profile_one(algorithm, program_arguments, compiler_flags, profile_misses);


            write_backup(backup_csv_filename);
    }

}

void Profiler::write_backup(const std::string &backup_filename) const {

    /*
     * This function writes the last line of the csv file into the backup file
     */


    //reading last line of csv file and writing it into backup file
    std::ifstream file(csv_filename);
    std::string last_line;
    std::string line;
    while (std::getline(file, line))
        last_line = line;

    std::ofstream backup_file;
    backup_file.open(backup_filename, std::ios_base::app); // opens file in append mode

    if (!backup_file.is_open()) {
        std::cerr << "Cannot open file: " << backup_filename << std::endl;
        return;
    }

    backup_file << last_line << std::endl;




}






