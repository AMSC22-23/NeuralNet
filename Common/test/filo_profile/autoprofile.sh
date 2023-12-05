#!/bin/bash

# Algorithms to test
algorithms=("algorithm1" "algorithm2" "algorithm3")

# Compiler Flags
optimization_levels=("-O3 -march=native -ffast-math" "Other opt_level")

optimization_ids = ("1")

# Defining output folder
output_folder="binaries"
mkdir -p $output_folder


for algo in "${algorithms[@]}"; do
  for opt_level in "${optimization_levels[@]}"; do
    # Composing executable name
    executable="$output_folder/$algo-$opt_level"

    # Compilazione dell'algoritmo con il livello di ottimizzazione corrente
    g++ -std=c++11 $opt_level $algo.cpp -o $executable

    # Esecuzione dello script Chrono e Valgrind
    ./measure.sh $executable # Assumendo che il tuo script di misurazione sia chiamato measure.sh
  done
done
