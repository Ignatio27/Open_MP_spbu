#!/usr/bin/env bash
set -e

SRC=${1:-results_extrema.csv}
DST=${2:-results_sorted.csv}

# Пересортировать по размеру и числу потоков (с сохранением заголовка)
{ head -n 1 "$SRC"; tail -n +2 "$SRC" | sort -t';' -k2,2n -k3,3n; } > "$DST"
echo "Отсортированная таблица записана в $DST"
