import time
import numpy as np
import os
import platform
import psutil
from multiprocessing import Pool, cpu_count, current_process
import threading

def system_info():
    print("==== SYSTEM INFO ====")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"CPU Cores: {cpu_count()} (logical)")
    print(f"Total RAM: {psutil.virtual_memory().total / 1e9:.2f} GB")
    print(f"Available RAM: {psutil.virtual_memory().available / 1e9:.2f} GB")
    print("=====================\n")

def print_thread_env_vars():
    print("BLAS / OpenMP environment variables:")
    for var in ["MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        print(f"  {var} = {os.environ.get(var, '(not set)')}")
    print()

def numpy_benchmark():
    print("Starting NumPy vectorized operation benchmark (sqrt + sum)...")
    start = time.time()
    arr = np.random.rand(10000, 10000)
    # Warm up once to load libraries
    np.sqrt(arr)
    start = time.time()
    result = np.sqrt(arr).sum()
    end = time.time()
    print(f"NumPy sqrt+sum took {end - start:.2f} seconds, result checksum: {result:.3f}\n")

def numpy_matmul_benchmark():
    print("Starting NumPy matrix multiplication benchmark (dot)...")
    arr = np.random.rand(2000, 2000)
    # Warm up once
    np.dot(arr, arr)
    start = time.time()
    result = np.dot(arr, arr)
    end = time.time()
    print(f"NumPy dot product took {end - start:.2f} seconds, result checksum: {result[0,0]:.3f}\n")

def multiprocessing_benchmark():
    print("Starting multiprocessing overhead benchmark...")
    nprocs = min(4, cpu_count())
    print(f"Using {nprocs} processes")
    start = time.time()
    with Pool(nprocs) as pool:
        results = pool.map(dummy_worker, range(nprocs * 5))
    end = time.time()
    print(f"Multiprocessing with {nprocs} processes took {end - start:.2f} seconds\n")

def dummy_worker(x):
    pid = os.getpid()
    p_name = current_process().name
    s = 0
    for i in range(100000):
        s += i**2
    print(f"Process {p_name} (PID: {pid}) finished task with x = {x}")
    return s + x

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
    print("Cleaning up...")
    for i in range(1000):
        os.remove(f"{folder}/file_{i}.txt")
    os.rmdir(folder)
    print("Done.\n")

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
    print("Cleaning up...")
    os.remove(big_file)
    os.rmdir(folder)
    print("Done.\n")

if __name__ == "__main__":
    system_info()
    print_thread_env_vars()
    numpy_benchmark()
    numpy_matmul_benchmark()
    multiprocessing_benchmark()
    small_file_write_test()
    big_file_write_test()
