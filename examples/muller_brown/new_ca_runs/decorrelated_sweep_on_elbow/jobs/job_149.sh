#!/bin/bash
#SBATCH --job-name=abc_149
#SBATCH --output=logs/abc_149.out
#SBATCH --error=logs/abc_149.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4310 3255 822 82 2855 2235 890 2539 2678 4054 3870 238 1257 1181 405 4155 1750 1843 4347 4815 3509 1068 327 4027 4209 2884
