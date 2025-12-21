#!/usr/bin/env bash
set -e

CXX=/opt/homebrew/bin/g++-15
CXXFLAGS="-O3 -std=c++17 -Wall -Wextra -fopenmp"

"$CXX" $CXXFLAGS -o dot_omp dot.cpp
