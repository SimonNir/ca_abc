#!/bin/bash
#SBATCH --job-name=abc_2
#SBATCH --output=logs/abc_2.out
#SBATCH --error=logs/abc_2.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 508 3996 2799 1672 2152 3652 347 3814 3642 3664 1863 295 2940 3276 1517 3479 2446 4883 1167 4266 4229 5091 1784 185 1653 1940
