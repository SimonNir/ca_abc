#!/bin/bash
#SBATCH --job-name=abc_81
#SBATCH --output=logs/abc_81.out
#SBATCH --error=logs/abc_81.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 2312 2229 1426 4009 2218 2255 2795 528 4457 2797 3808 2271 3090 2387 2751 298 2775 3749 4249 256 1392 1460 2609 2963 1567 1449
