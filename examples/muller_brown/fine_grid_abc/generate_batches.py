import os, json, random
from itertools import product

RUNS_PER_BATCH = 10
RESULT_DIR = "results"
BATCH_DIR = "batches"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(BATCH_DIR, exist_ok=True)

def get_all_params():
    std_devs = [1/3, 1/5, 1/8, 1/10, 1/14]
    heights = [1/5, 1/10, 1/30, 1/50, 1/100]
    perturb = [0.005]
    optims = [0]
    iters = 10
    return [(i, *p) for i, p in enumerate(product(std_devs, heights, perturb, optims, iters))]

def get_completed():
    done = set()
    for f in os.listdir(RESULT_DIR):
        if not f.startswith("batch_") or not f.endswith(".json"):
            continue
        with open(os.path.join(RESULT_DIR, f)) as jf:
            for entry in json.load(jf):
                done.add(entry["run_id"])
    return done

def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

def main():
    all_params = get_all_params()
    done = get_completed()
    pending = [p for p in all_params if p[0] not in done]

    random.shuffle(pending)
    batches = list(chunk(pending, RUNS_PER_BATCH))

    for i, batch in enumerate(batches):
        with open(f"{BATCH_DIR}/batch_{i}.json", "w") as f:
            json.dump(batch, f)

    with open("submit_all.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name=abc\n#SBATCH --output=logs/slurm_%A_%a.out\n")
        f.write("#SBATCH --error=logs/slurm_%A_%a.err\n#SBATCH --array=0-{}\n".format(len(batches)-1))
        f.write("#SBATCH --time=01:00:00\n#SBATCH --mem=4G\n#SBATCH --cpus-per-task=1\n\n")
        f.write("source ~/.bashrc\nconda activate your_env_name\n")
        f.write("python run_batch.py batches/batch_${SLURM_ARRAY_TASK_ID}.json\n")

    print(f"Generated {len(batches)} batch files and SLURM script.")

if __name__ == "__main__":
    main()
