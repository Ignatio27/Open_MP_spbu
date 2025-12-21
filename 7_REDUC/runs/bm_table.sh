#!/usr/bin/env bash
set -e

SRC="reduce_results.csv"
OUT="reduce_mutex_bench.csv"

# просто копируем и оставляем только нужные столбцы / строки при желании
cp "${SRC}" "${OUT}"
