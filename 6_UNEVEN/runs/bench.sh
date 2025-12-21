#!/usr/bin/env bash
set -e

BIN=./loop_schedules

$BIN --benchmark --n 500000  > bench_500k.csv
$BIN --benchmark --n 5000000 > bench_5m.csv
