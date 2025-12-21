#!/usr/bin/env bash
set -e

CXX=/opt/homebrew/bin/g++-15
CXXFLAGS="-O3 -std=c++17 -fopenmp -Wall -Wextra -march=native"

$CXX $CXXFLAGS -o find_extrema main.cpp
