# !/bin/bash
# This script runs the QRL-Pong experiments

uv sync
uv run benchmarkClassicalBaseline64P.py
uv run benchmarkClassicalBaseline4096P.py
uv run main.py