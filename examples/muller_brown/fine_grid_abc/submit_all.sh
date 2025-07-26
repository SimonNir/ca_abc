#!/bin/bash
mkdir -p jobs logs

# Get all remaining run_ids
remaining_ids=$(python -c "
from sweep import get_all_run_params, get_completed_runs, RESULT_DIR
all_runs = get_all_run_params()
completed = get_completed_runs(RESULT_DIR)
remaining = [str(r[0]) for r in all_runs if r[0] not in completed]
print(' '.join(remaining))
")

# Split into chunks for each core
NUM_CORES=100
readarray -t id_array <<< "$(echo $remaining_ids | tr ' ' '\n' | shuf)"
ids_per_core=$(( (${#id_array[@]} + NUM_CORES - 1) / NUM_CORES ))

for core_id in $(seq 0 $((NUM_CORES - 1))); do
    start=$((core_id * ids_per_core))
    end=$((start + ids_per_core))
    core_run_ids="${id_array[@]:start:ids_per_core}"
    
    if [ -z "$core_run_ids" ]; then
        continue
    fi

    script="jobs/job_${core_id}.sh"
    cat <<EOF > $script
#!/bin/bash
#SBATCH --job-name=abc_${core_id}
#SBATCH --output=logs/abc_${core_id}.out
#SBATCH --error=logs/abc_${core_id}.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --mem=2G
#SBATCH -p burst
#SBATCH -A birthright

source ~/abc_venv/bin/activate
export PYTHONPATH=\$PYTHONPATH:\$(pwd)
python run_one.py ${core_run_ids}
EOF

    chmod +x $script
    sbatch $script
done