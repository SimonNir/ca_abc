import sys, json
from run_utils import single_run, convert_numpy

def main():
    batch_file = sys.argv[1]
    batch_id = batch_file.split("_")[-1].split(".")[0]

    with open(batch_file) as f:
        batch = json.load(f)

    results = []
    for args in batch:
        run_data = single_run(args)
        if run_data:
            results.append(run_data)

    with open(f"results/batch_{batch_id}.json", "w") as f:
        json.dump(convert_numpy(results), f)

if __name__ == "__main__":
    main()
