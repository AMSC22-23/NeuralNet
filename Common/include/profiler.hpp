#include <vector>
#include <string>
#include <map>

/*
 *  To compile:
 *
 *  g++ test_profiler.cpp ../../src/profiler.cpp ../../src/mmm.cpp -o autoprofile
 */

#ifndef SHAREDFOLDER_PROFILER_HPP
#define SHAREDFOLDER_PROFILER_HPP


class Profiler{

private:

    const std::string cachegrind_log_filename;
    const std::string csv_filename;
    const std::string profile_list_filename;
    const std::string linker_flags = "-I${mkOpenblasInc} -L${mkOpenblasLib} -lopenblas";


    static std::string format4csv(const std::string& input_string);
    void append_misses(const std::string& misses) const;
    void cachegrindReader() const ;

    void profile_one(const std::string& compiler_flags, const std::string& algorithm, const std::string& program_arguments) const;


public:


    Profiler(const std::string& log_filename, const std::string& csv_name, const std::string& profile_list_name):
            cachegrind_log_filename(log_filename),
            csv_filename(csv_name),
            profile_list_filename(profile_list_name)
            {};


    void profile() const;

};



#endif