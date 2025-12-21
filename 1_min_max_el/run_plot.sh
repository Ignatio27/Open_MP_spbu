#!/usr/bin/env bash
set -e

INPUT_CSV=${1:-bm_table.csv}

python3 draw_extrema_plots.py "$INPUT_CSV"
