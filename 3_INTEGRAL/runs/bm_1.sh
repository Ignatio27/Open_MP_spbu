#!/usr/bin/env bash
set -e

BIN=./integral_omp

"$BIN" --func sin --rule mid --a 0 --b 3.1415926535 --N 5000000 --threads 8
