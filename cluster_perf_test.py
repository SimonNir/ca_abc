import time
import numpy as np
import os
from multiprocessing import Pool, cpu_count

def numpy_benchmark():
    print("Starting NumPy vectorized operation benchmark...")
    start = time.time()
    arr = np.random.rand(10000, 10000)
    result = np.sqrt(arr).sum()
    end = time.time()
    print(f"NumPy operation took {end - start:.2f} seconds")

def dummy_worker(x):
    # Simple CPU-bound task
    s = 0
    for i in range(100000):
        s += i**2
    return s + x

def multiprocessing_benchmark():
    print("Starting multiprocessing overhead benchmark...")
    nprocs = min(4, cpu_count())
    start = time.time()
    with Pool(nprocs) as pool:
        results = pool.map(dummy_worker, range(nprocs*5))
    end = time.time()
    print(f"Multiprocessing with {nprocs} processes took {end - start:.2f} seconds")

def small_file_write_test():
    print("Starting small file write test (1000 files)...")
    folder = "tmp_small_files"
    os.makedirs(folder, exist_ok=True)
    start = time.time()
    for i in range(1000):
        with open(f"{folder}/file_{i}.txt", "w") as f:
            f.write("Hello world!\n" * 10)
    end = time.time()
    print(f"Wrote 1000 small files in {end - start:.2f} seconds")
    # Clean up
    for i in range(1000):
        os.remove(f"{folder}/file_{i}.txt")
    os.rmdir(folder)

def big_file_write_test():
    print("Starting big file write test (1 file)...")
    folder = "tmp_big_file"
    os.makedirs(folder, exist_ok=True)
    big_file = f"{folder}/big_file.txt"
    start = time.time()
    with open(big_file, "w") as f:
        for _ in range(100000):
            f.write("Hello world! " * 10 + "\n")
    end = time.time()
    print(f"Wrote 1 big file in {end - start:.2f} seconds")
    os.remove(big_file)
    os.rmdir(folder)

if __name__ == "__main__":
    numpy_benchmark()
    multiprocessing_benchmark()
    small_file_write_test()
    big_file_write_test()
