#!/bin/bash
#SBATCH --job-name=abc_0
#SBATCH --output=logs/abc_0.out
#SBATCH --error=logs/abc_0.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=3G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

python run_one.py 4287 4236 3981 4922 812 4524 4412 2961 1735 4841 3750 4420 2165 5010 4120 3842 573 4435 3820 4706 3111 1952 2417 4578 4721 698
