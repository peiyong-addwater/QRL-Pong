#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --mem=5GB
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --nodes=1
#SBATCH --account=OD-231649
#SBATCH --ntasks-per-node=1 
#SBATCH --mail-type=ALL
#SBATCH --output="../sbatch_log/%A.out"

HOME="/scratch3/ip004"
VENV="../venv"
export HF_HOME="$HOME/.cache/HuggingFace"
export PIP_CACHE_DIR="$HOME/.cache/pip"

# pyenv local 3.12
python3 -m venv $VENV
source $VENV/bin/activate

python util.py