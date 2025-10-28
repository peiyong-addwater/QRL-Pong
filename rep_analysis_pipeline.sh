# !/bin/bash
# Backbone representation analysis pipeline

uv sync

uv run obs_collection.py
uv run get_reps.py
uv run rep_sim_analysis.py
uv run plot_csv.py