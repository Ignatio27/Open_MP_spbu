#!/usr/bin/env bash
set -e

CXX=${CXX:-/opt/homebrew/bin/g++-15}
CXXFLAGS="-O3 -std=c++17 -fopenmp -Wall -Wextra -pedantic"

SRC="nested_rowminmax.cpp"
OUT="nested_rowminmax"

echo "Компиляция $SRC -> $OUT"
"$CXX" $CXXFLAGS -o "$OUT" "$SRC"
echo "Готово."
