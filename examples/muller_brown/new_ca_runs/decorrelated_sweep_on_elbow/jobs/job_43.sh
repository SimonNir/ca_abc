#!/bin/bash
#SBATCH --job-name=abc_43
#SBATCH --output=logs/abc_43.out
#SBATCH --error=logs/abc_43.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 3386 3072 2935 406 4844 407 3802 1225 420 4906 1694 2081 894 1398 2890 28 4048 971 2441 3857 1968 1808 1 4429 3050 4970
