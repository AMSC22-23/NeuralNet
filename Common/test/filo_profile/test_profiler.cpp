#include "../../include/profiler.hpp"

int main(){

    Profiler p("TEMP.txt", "filResult.csv", "profile_list.txt");
    p.profile();

    return 0;
}