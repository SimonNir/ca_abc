import numpy as np
import time 


# Generate a random invertible 100x100 matrix
while True:
    H = np.random.rand(100, 100)
    if np.linalg.matrix_rank(H) == 100:
        break
print("generated")
H_0 = H.copy()
start = time.time()
for i in range(1000):
    H = np.linalg.inv(H)
end = time.time()

print(end-start)
print(np.sum(np.abs(H-H_0)))

# this took 46 seconds on my computer. It's not worth fixing the code to have fewer inversions