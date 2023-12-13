#include "../../include/profiler.hpp"

int main(){

    Profiler p("TEMP.txt", "profiling_results.csv", "profile_list.txt");
    p.profile();

    return 0;
}