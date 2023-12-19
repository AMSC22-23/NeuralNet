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
    /*
     * This class is used to profile different algorithms. It allows to measure time complexity and cache misses of
     * different matrix multiplication algorithms, when different parameters are used.
     * This class profiles the algorithms listed in the file profile_list_filename, and writes the results in the file
     * csv_filename and backup_csv_filename.
     * In the profile_list_filename file, each line represents a different profiling where different profiling can be specified,
     * and has the following format:
     * algorithm_name,program_arguments,compiler_flags,profile_misses
     *
     * We used this class to tests different compiler flags, different algorithms, and how performance changes when
     * different parameters are used.
     *
     * For making an algorithm profilable with this class it is necessary that in the folder where the program is executed
     * there is a file called ALGORITHM_NAME.cpp, which contains the function that implements the algorithm.
     *
     * The output of the profiling is written in both csv_filename and backup_csv_filename files.
     * - backup_csv_filename contains all the profiling results.
     * - csv_filename contains profiling results that are not already uploaded to the mongodb database.
     *   When new plot are generated, the profiling results contained in this csv file are uploaded to the database
     *   and the content of the csv file is deleted, to avoid uploading the same profiling results twice.
     *
     * The upload of data to the mongodb database is not manged by this class, but by the python script dbconnection.py,
     * executed only when new plots are needed.
     * The choice of mongodb is due to the fact that output data may not be structured in a tabular format,
     * (even thought right now it is), and its environment was already installed on the machine where the profiling was executed.
     */

private:

    const std::string cachegrind_log_filename;   // a temp file used to store the output of valgrind
    const std::string csv_filename;       // the file where the profiling results are written and then cancelled
    const std::string backup_csv_filename = "filResult.csv"; // the file where the profiling results are written and never cancelled
    const std::string profile_list_filename; // the file where the algorithms to profile are listed
    const std::string linker_flags = "-I${mkOpenblasInc} -L${mkOpenblasLib} -lopenblas"; // the flags used to link openblas



    static std::string format4csv(const std::string& input_string);
    void append_misses(const std::string& misses) const;
    void cachegrindReader() const ;

    void profile_one(const std::string& compiler_flags, const std::string& algorithm, const std::string& program_arguments,
                            bool profile_misses = false) const;

    void write_backup(const std::string& backup_filename) const;


public:


    Profiler(const std::string& log_filename, const std::string& csv_name, const std::string& profile_list_name):
            cachegrind_log_filename(log_filename),
            csv_filename(csv_name),
            profile_list_filename(profile_list_name)
            {};


    void profile() const;


};



#endif