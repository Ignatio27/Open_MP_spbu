#!/usr/bin/env bash
set -e

CXX=${CXX:-/opt/homebrew/bin/g++-15}
CXXFLAGS="-O3 -std=c++17 -fopenmp -Wall -Wextra -pedantic"

$CXX $CXXFLAGS -o reduce_bench reduce_experiment.cpp
