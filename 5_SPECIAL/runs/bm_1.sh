#!/usr/bin/env bash
set -e

# одиночный прогон для примера:
# нижнетреугольная матрица, 20000×20000, dynamic, 4 потока
./matrix_game \
  --rows 20000 --cols 20000 \
  --kind lower \
  --threads 4 \
  --sched dynamic
