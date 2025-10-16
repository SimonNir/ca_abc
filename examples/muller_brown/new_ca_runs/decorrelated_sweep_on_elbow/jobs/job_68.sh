#!/bin/bash
#SBATCH --job-name=abc_68
#SBATCH --output=logs/abc_68.out
#SBATCH --error=logs/abc_68.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4160 2993 4346 5054 1003 1979 3885 924 2760 1153 562 3421 642 158 3155 2659 4995 2241 1547 1232 3603 4937 4941 598 2597 2826
