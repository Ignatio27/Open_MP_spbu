#!/usr/bin/env bash
set -euo pipefail

CSV=${1:-nested_bench.csv}
OUT=${2:-nested_bench_head.csv}

# просто берём первые строки для быстрого просмотра
head -n 40 "${CSV}" > "${OUT}"
echo "Фрагмент таблицы записан в ${OUT}"
