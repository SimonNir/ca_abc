#!/bin/bash
#SBATCH --job-name=abc_49
#SBATCH --output=logs/abc_49.out
#SBATCH --error=logs/abc_49.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2135 2448 1205 3769 4682 1837 5103 3202 2969 1206 2313 4222 160 883 1314 5099 2464 2891 2232 2943 3716 4779 426 4623 2647 1768
