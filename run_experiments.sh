# !/bin/bash
# This script runs the QRL-Pong experiments

uv sync
uv run benchmarkClassicalBaseline.py
uv run main.py