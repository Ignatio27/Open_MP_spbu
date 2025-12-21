#!/usr/bin/env bash
set -euo pipefail

BIN=./nested_rowminmax
OUT_CSV=nested_bench.csv

echo "mode,nrows,ncols,outer_threads,inner_threads,outer_sched,inner_sched,outer_chunk,inner_chunk,time_sec,speedup_vs_seq" > "$OUT_CSV"

M_SIZES=(500)
TOTAL_THREADS=(1)
SCHEDULES=("static")

run_outer() {
    local n=$1
    local total=1

    for threads in "${TOTAL_THREADS[@]}"; do
        for sched in "${SCHEDULES[@]}"; do
            for ((i=0; i<total; ++i)); do
                $BIN --bench --mode outer \
                     --nrows "$n" --ncols "$n" \
                     --threads "$threads" --sched "$sched" \
                >> "$OUT_CSV"
            done
        done
    done
}

run_outer 500
