#!/usr/bin/env bash
set -e

CXX=${CXX:-/opt/homebrew/bin/g++-15}
FLAGS="-O3 -std=c++17 -fopenmp -Wall -Wextra -pedantic"

$CXX $FLAGS -o matrix_game solver_special.cpp
