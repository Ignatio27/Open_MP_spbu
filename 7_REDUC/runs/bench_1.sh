#!/usr/bin/env bash
set -e

BIN=./reduce_bench

N_VALUES=("100000" "1000000" "5000000" "20000000")
THREADS=("1" "2" "4" "5" "8")
METHODS=("atomic" "critical" "lock" "reduction")

OUT_CSV="reduce_results.csv"
echo "n,threads,method,time_sec" > "${OUT_CSV}"

for n in "${N_VALUES[@]}"; do
  for t in "${THREADS[@]}"; do
    for m in "${METHODS[@]}"; do
      ${BIN} --n "${n}" --threads "${t}" --mode "${m}" >> "${OUT_CSV}"
    done
  done
done