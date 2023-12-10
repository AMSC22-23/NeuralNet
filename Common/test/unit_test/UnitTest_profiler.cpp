#include "../../include/profiler.hpp"

int main(){

    Profiler p("TEMP.txt", "filResult.csv");
    p.cachegrindReader();

    return 0;
}