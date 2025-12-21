#!/usr/bin/env bash
set -e

BIN=./matrix_game

# Треугольные и ленточная
$BIN --bench --kind lower > bench_lower.csv
$BIN --bench --kind upper > bench_upper.csv
$BIN --bench --kind band --bw 64 > bench_band.csv
$BIN --bench --kind dense > bench_full.csv
