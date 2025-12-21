#!/usr/bin/env bash
set -e

BIN=./maximin_matrix
OUT=bm_table.csv

"$BIN" --bench > "$OUT"
