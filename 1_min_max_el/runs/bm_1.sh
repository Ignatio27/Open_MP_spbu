#!/usr/bin/env bash
set -e

OUT_FILE=${1:-results_extrema.csv}

./find_extrema -x > "$OUT_FILE"
echo "CSV сохранён в $OUT_FILE"
