#!/usr/bin/env bash
set -euo pipefail
# FASTLANE quick repro: compile & run (adjust -arch as needed)
nvcc -O3 -std=c++17 -arch=sm_80 fastlane_best_rr_allinone.cu -lcublasLt -lcublas -o fastlane_best_rr_allinone
./fastlane_best_rr_allinone
