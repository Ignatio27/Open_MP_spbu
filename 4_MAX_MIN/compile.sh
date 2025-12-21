#!/usr/bin/env bash
set -e

CXX=${CXX:-/opt/homebrew/bin/g++-15}
CXXFLAGS="-O3 -std=c++17 -Wall -Wextra -pedantic -fopenmp"

"$CXX" $CXXFLAGS -o maximin_matrix maximin_matrix.cpp
