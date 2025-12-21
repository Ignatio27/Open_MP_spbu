#!/usr/bin/env bash
set -e

BIN=./integral_omp
OUT=bm_table.csv

"$BIN" --bench > "$OUT"
