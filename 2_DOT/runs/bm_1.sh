#!/usr/bin/env bash
set -e

./dot_omp --n 5000000 --threads 4 --mode red
./dot_omp --n 5000000 --threads 4 --mode acc
